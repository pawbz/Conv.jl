__precompile__()

"""
This module is used in DeConv, ConvMix and Coupling
"""
module Conv
using FFTW

using DSP.nextfastfft

"""
Model : d = convolution(g, s)
d, g and s can have arbitary +ve and -ve lags
"""
type Param{T<:Real, Nd, Ng, Ns}
	dsize::Vector{Int}
	gsize::Vector{Int}
	ssize::Vector{Int}
	"length after zero padding"
	np2::Int # nfastfft length
	d::Array{T,Nd}
	g::Array{T,Ng}
	s::Array{T,Ns}
	dpad::Array{T,Nd}
	gpad::Array{T,Ng}
	spad::Array{T,Ns}
	dfreq::Array{Complex{T},Nd}
	gfreq::Array{Complex{T},Ng}
	sfreq::Array{Complex{T},Ns}
	dfftp::FFTW.rFFTWPlan
	difftp::FFTW.Plan
	gfftp::FFTW.rFFTWPlan
	gifftp::FFTW.Plan
	sfftp::FFTW.rFFTWPlan
	sifftp::FFTW.Plan
	"+ve and -ve lags of g"
	glags::Vector{Int}
	"+ve and -ve lags of s"
	slags::Vector{Int}
	"+ve and -ve lags of d"
	dlags::Vector{Int}
	dsum::Bool
	gsum::Bool
	ssum::Bool
end

function Base.fill!(pa::Param, k)
	for iff in [:d, :g, :s]
		m=getfield(pa,iff)
		for i in eachindex(m)
			m[i]=eltype(pa.d)(k)
		end
	end
end



function Param(; 
	       dsize::Vector{Int64}=[1],
	       ssize::Vector{Int64}=[1],
	       gsize::Vector{Int64}=[1],
	       d=zeros(dsize...), 
	       s=zeros(ssize...),
	       g=zeros(gsize...),
	       slags=nothing, 
	       dlags=nothing, 
	       glags=nothing,
	       np2=nextfastfft(maximum([2*dsize[1], 2*gsize[1], 2*ssize[1]])), # fft dimension for plan
	       fftwflag=FFTW.ESTIMATE
	      )
	Nd=length(dsize)
	Ns=length(ssize)
	Ng=length(gsize)
	nts=ssize[1]
	ntg=gsize[1]
	ntd=dsize[1]

	(eltype(d) ≠ eltype(s) ≠ eltype(g)) && error("type error")
	T=eltype(d)


	# default lags
	if(slags===nothing)
		# equal +ve and -ve lags for s
		nwplags=div(nts-1,2)
			nwnlags=nts-1-nwplags
		slags=[nwplags, nwnlags]
	end
	if(dlags===nothing)
		# no negative lags
		nsplags=ntd-1
		nsnlags=ntd-1-nsplags
		dlags=[nsplags, nsnlags]
	end
	if(glags===nothing)
		# no negative lags
		nrplags=ntg-1
		nrnlags=ntg-1-nrplags
		glags=[nrplags, nrnlags]
	end


	# check if nlags are consistent with the first dimension of inputs
	(sum(glags)+1 ≠ size(g,1)) && error("glags")
	(sum(dlags)+1 ≠ size(d,1)) && error("dlags")
	(sum(slags)+1 ≠ size(s,1)) && error("slags")

	FFTW.set_num_threads(Sys.CPU_CORES)
	nrfft=div(np2,2)+1


	dfftp=plan_rfft(zeros(T, np2, dsize[2:end]...),[1], flags=fftwflag,  timelimit=Inf)
	difftp=plan_irfft(complex.(zeros(T, nrfft, dsize[2:end]...)),np2,[1], flags=fftwflag,  timelimit=Inf)

	if(gsize[2:end] ≠  dsize[2:end])
		gfftp=plan_rfft(zeros(T, np2, gsize[2:end]...),[1], flags=fftwflag,  timelimit=Inf)
		gifftp=plan_irfft(complex.(zeros(T, nrfft, gsize[2:end]...)),np2,[1], flags=fftwflag,  timelimit=Inf)
	else
		gfftp=dfftp
		gifftp=difftp
	end

	if(ssize[2:end] ≠ dsize[2:end])
		sfftp=plan_rfft(zeros(T, np2, ssize[2:end]...),[1], flags=fftwflag,  timelimit=Inf)
		sifftp=plan_irfft(complex.(zeros(T, nrfft, ssize[2:end]...)),np2,[1], flags=fftwflag,  timelimit=Inf)
	else
		sfftp=dfftp
		sifftp=difftp

	end


	# preallocate freq domain vectors after rfft
	dfreq=complex.(zeros(T,nrfft,dsize[2:end]...))
	gfreq=complex.(zeros(T,nrfft,gsize[2:end]...))
	sfreq=complex.(zeros(T,nrfft,ssize[2:end]...))

	# preallocate padded arrays
	dpad=(zeros(T,np2,dsize[2:end]...))
	gpad=(zeros(T,np2,gsize[2:end]...))
	spad=(zeros(T,np2,ssize[2:end]...))

	dsum=false; ssum=false; gsum=false
	if(!(dsize[2:end]==ssize[2:end]==gsize[2:end]))
		if(Nd==1)
			dsum=true
			(gsize[2:end]≠ssize[2:end]) && error("dimension error")
		elseif(Ng==1) 
			gsum=true
			(dsize[2:end]≠ssize[2:end]) && error("dimension error")
		elseif(Ns==1)
			ssum=true
			(dsize[2:end]≠gsize[2:end]) && error("dimension error")
		else
			error("only vectors allowed, otherwise g, s, d should have same size")
		end
	end

	pa=Param(dsize, gsize, ssize, np2, d, g, s, 
	  dpad, gpad, spad,
	  dfreq, gfreq, sfreq, 
	  dfftp, difftp, 
	  gfftp, gifftp, 
	  sfftp, sifftp, 
	  glags, slags, dlags, dsum, gsum, ssum)
end


"""
Convolution that allocates `Param` internally.
Don,t use inside loops
"""
function mod!{T,Nd,Ng,Ns}(
	   d::AbstractArray{T,Nd}, 
	   g::AbstractArray{T,Ng},
	   s::AbstractArray{T,Ns}, attrib::Symbol)
	(attrib ∉ [:s, :d, :g]) && error("invalid attrib")
	dsize=[size(d)...]
	gsize=[size(g)...]
	ssize=[size(s)...]

	# allocation of freq matrices
	pa=Param(dsize=dsize, ssize=ssize, gsize=gsize, g=g, s=s, d=d)

	# using pa, return d, g, s according to attrib
	mod!(pa, attrib)
end

"""
Convolution modelling with no allocations at all.
By default, the fields `g`, `d` and `s` in pa are modified accordingly.
Otherwise use keyword arguments to input them.
"""
function mod!(pa::Param, attrib::Symbol; 
	      g=pa.g, d=pa.d, s=pa.s # external arrays to be modified
	     )

	(attrib ∉ [:s, :d, :g]) && error("invalid attrib")
	T=eltype(pa.d)
	
	# initialize freq vectors
	pa.dfreq[:] = complex(T(0))
	pa.gfreq[:] = complex(T(0))
	pa.sfreq[:] = complex(T(0))

	pa.gpad[:]=T(0)
	pa.dpad[:]=T(0)
	pa.spad[:]=T(0)

	# necessary zero padding
	pad_truncate!(g, pa.gpad, pa.glags[1], pa.glags[2], pa.np2, 1)
	pad_truncate!(d, pa.dpad, pa.dlags[1], pa.dlags[2], pa.np2, 1)
	pad_truncate!(s, pa.spad, pa.slags[1], pa.slags[2], pa.np2, 1)

	if(attrib == :d)
		A_mul_B!(pa.sfreq, pa.sfftp, pa.spad)
		A_mul_B!(pa.gfreq, pa.gfftp, pa.gpad)
		if(pa.dsum)
			for i in CartesianRange(size(pa.gfreq))
				pa.dfreq[i[1]] += pa.gfreq[i] * pa.sfreq[i]
			end
		elseif(pa.ssum)
			for i in CartesianRange(size(pa.gfreq))
				pa.dfreq[i] = pa.gfreq[i] * pa.sfreq[i[1]]
			end
		elseif(pa.gsum)
			for i in CartesianRange(size(pa.sfreq))
				pa.dfreq[i] = pa.gfreq[i[1]] * pa.sfreq[i]
			end
		else
			for i in eachindex(pa.dfreq)
				pa.dfreq[i] = pa.gfreq[i] * pa.sfreq[i]
			end
		end
		A_mul_B!(pa.dpad, pa.difftp, pa.dfreq)
		pad_truncate!(d, pa.dpad, pa.dlags[1], pa.dlags[2], pa.np2, -1)
		return d
	elseif(attrib == :g)
		A_mul_B!(pa.sfreq, pa.sfftp, pa.spad)
		A_mul_B!(pa.dfreq, pa.dfftp, pa.dpad)
		conj!(pa.sfreq)
		if(pa.dsum)
			for i in CartesianRange(size(pa.gfreq))
				pa.gfreq[i] = pa.dfreq[i[1]] * pa.sfreq[i]
			end
		elseif(pa.ssum)
			for i in CartesianRange(size(pa.gfreq))
				pa.gfreq[i] = pa.dfreq[i] * pa.sfreq[i[1]]
			end
		elseif(pa.gsum)
			for i in CartesianRange(size(pa.sfreq))
				pa.gfreq[i[1]] += pa.dfreq[i] * pa.sfreq[i]
			end
		else
			for i in eachindex(pa.dfreq)
				pa.gfreq[i] = pa.dfreq[i] * pa.sfreq[i]
			end
		end
#		@. pa.gfreq = pa.dfreq * pa.sfreq
		A_mul_B!(pa.gpad, pa.gifftp, pa.gfreq)
		pad_truncate!(g, pa.gpad, pa.glags[1], pa.glags[2], pa.np2, -1)
		
		return g
	elseif(attrib == :s)
		A_mul_B!(pa.gfreq, pa.gfftp, pa.gpad)
		A_mul_B!(pa.dfreq, pa.dfftp, pa.dpad)
		conj!(pa.gfreq)
		if(pa.dsum)
			for i in CartesianRange(size(pa.gfreq))
				pa.sfreq[i] = pa.dfreq[i[1]] * pa.gfreq[i]
			end
		elseif(pa.ssum)
			for i in CartesianRange(size(pa.gfreq))
				pa.sfreq[i[1]] += pa.dfreq[i] * pa.gfreq[i]
			end
		elseif(pa.gsum)
			for i in CartesianRange(size(pa.sfreq))
				pa.sfreq[i] = pa.dfreq[i] * pa.gfreq[i[1]]
			end
		else
			for i in eachindex(pa.dfreq)
				pa.sfreq[i] = pa.dfreq[i] * pa.gfreq[i]
			end
		end
		#@. pa.sfreq = pa.dfreq * pa.gfreq
		A_mul_B!(pa.spad, pa.sifftp, pa.sfreq)
		pad_truncate!(s, pa.spad, pa.slags[1], pa.slags[2], pa.np2, -1)
		return s
	end
end


"""
Method to perform zero padding and truncation.

# Arguments

* `x` : real signal with dimension nplag + nnlag + 1
	first has decreasing negative lags, 
	then has zero lag at nnlags + 1,
	then has increasing positive nplags lags,
	signal contains only positive lags and zero lag if nnlag=0 and vice versa
* `xpow2` : npow2 real vector with dimension npow2
* `nplags` : number of positive lags
* `nnlags` : number of negative lags
* `npow2` : number of samples in xpow2
* `flag` : = 1 means xpow2 is returned using x
	   = -1 means x is returned using xpow2
"""
function pad_truncate!{T}(
				  x::AbstractArray{T}, 
				  xpow2::AbstractArray{T}, 
				  nplags::Integer, 
				  nnlags::Integer, 
				  npow2::Integer, 
				  flag::Integer
				  )
	(size(x,1) ≠ nplags + nnlags + 1) && error("size x")
	(size(xpow2,1) ≠ npow2) && error("size xpow2")

	for id in 1:size(x,2)
		if(flag == 1)
			xpow2[1,id] = (x[nnlags+1,id]) # zero lag
			# +ve lags
			if (nplags > 0) 
				for i=1:nplags
					xpow2[i+1,id]= (x[nnlags+1+i,id])
				end
			end
			# -ve lags
			if(nnlags != 0) 
				for i=1:nnlags
					xpow2[npow2-i+1,id] =(x[nnlags+1-i,id])
				end
			end
		elseif(flag == -1)
			x[nnlags+1,id] = (xpow2[1,id]); # zero lag
			if(nplags != 0) 
				for i=1:nplags
					x[nnlags+1+i,id] = (xpow2[1+i,id]);
				end
			end
			if(nnlags != 0)
				for i=1:nnlags
					x[nnlags+1-i,id] = (xpow2[npow2-i+1,id])
				end
			end
		else
			error("invalid flag")
		end
	end
	return nothing
end


"""
Parameters to generate all possible cross-correlations
"""
mutable struct Param_xcorr
	paconv::Conv.Param{Float64,1}
	iref::Vector{Int64}
	norm_flag::Bool
end

function Param_xcorr(nt::Int64, iref, nts::Int64=2*nt-1, lags=nothing; norm_flag=true)
	(lags===nothing) && (lags=[nt-1, nt-1])
	s=zeros(nts);
	g=zeros(nt);
	d=zeros(nt);
	paconv=Param(gsize=[nt], dsize=[nt], ssize=[nts], g=g, s=s, d=d, slags=lags)
	pa=Param_xcorr(paconv, collect(iref), norm_flag)
	return pa
end

function xcorr(A::AbstractArray{Float64}; lags=[size(A,1)-1, size(A,1)-1], iref=0, norm_flag=true)
	nr=size(A,2)
	if(iref==0)
		iref=1:nr
	end
	Ax=[zeros(sum(lags)+1, nr-iref[i]+1) for i in 1:length(iref)]

	nt=size(A,1)
	nts=sum(lags)+1
	pa=Param_xcorr(nt, iref, nts, lags, norm_flag=norm_flag)

	return xcorr!(Ax, A, pa)
end


"""
Use first colomn of A and cross-correlate with rest of columns of it.
And store results in Ax.
After xcorr, the coeffcients are normalized with norm(A[:,1])
By default, Ax has almost same positive and negative lags.
"""
function xcorr!(Ax, A::AbstractArray{Float64}, pa)

	nr=size(A,2)
	iref=pa.iref
	lags=pa.paconv.slags
	norm_flag=pa.norm_flag

	any([(size(Ax[ir]) ≠ (sum(lags)+1,nr-ir+1)) for ir in pa.iref]) && error("size Ax")


	irrr=0
	α=0.0
	nt=size(A,1)
	nts=sum(lags)+1
	ir1=0
	for ir in iref
		ir1+=1
		Axx=Ax[ir1]
		for i in 1:nt
			pa.paconv.d[i]=A[i,ir]
		end
		if(ir1==1 && norm_flag) # save scale reference for first column only 
			for i in 1:nt
				α+=A[i,ir]*A[i,ir]
			end
			α = (iszero(α)) ? 1.0 : inv(α) # take inverse if not zero
		end
		for (iir2,ir2) in enumerate(ir:nr)
			for i in 1:nt
				pa.paconv.g[i]=A[i,ir2]
			end
			mod!(pa.paconv, :s)
			for i in 1:nts
				Axx[i,iir2]=pa.paconv.s[i]
			end
			if(norm_flag) 
				for i in 1:nts
					Axx[i,iir2]/=α
				end
			end
		end
	end
	return Ax
end

"""
given dJdAx and A, computes dJdA
"""
function xcorr_grad!(dA::AbstractArray{Float64}, dAx, A::AbstractArray{Float64}, pa)
	nr=size(A,2)
	nt=size(A,1)
	iref=pa.iref
	lags=pa.paconv.slags
	nts=sum(lags)+1
	dA[:]=0.0

	ir1=0
	for ir in iref
		ir1+=1
		dAxx=dAx[ir1]

		for ir2 in 1:nr

			for i in 1:nt
				pa.paconv.g[i]=A[i,ir2]
			end
			for i in 1:nts
				pa.paconv.s[i]=dAxx[i,ir2]
			end

			mod!(pa.paconv, :d) # check

			for i in 1:nt
				dA[i,ir]+=pa.paconv.d[i]
			end
		end

		for i in 1:nt
			pa.paconv.d[i]=A[i,ir]
		end

		for ir2 in 1:nr
			for i in 1:nts
				pa.paconv.s[i]=dAxx[i,ir2]
			end

			mod!(pa.paconv, :g) # check
			for i in 1:nt
				dA[i,ir2] +=pa.paconv.g[i]
			end
		end
	end
	return dA

end

end
