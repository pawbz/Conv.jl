__precompile__()

"""
This module is used in DeConv, ConvMix and Coupling
"""
module Conv
using FFTW

using DSP.nextfastfft
using Misfits

"""
Model : d = convolution(g, s)
d, g and s can have arbitary +ve and -ve lags
"""
type P_conv{T<:Real, Nd, Ng, Ns}
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

function Base.fill!(pa::P_conv, k)
	for iff in [:d, :g, :s]
		m=getfield(pa,iff)
		for i in eachindex(m)
			m[i]=eltype(pa.d)(k)
		end
	end
end



function P_conv(; 
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

	pa=P_conv(dsize, gsize, ssize, np2, d, g, s, 
	  dpad, gpad, spad,
	  dfreq, gfreq, sfreq, 
	  dfftp, difftp, 
	  gfftp, gifftp, 
	  sfftp, sifftp, 
	  glags, slags, dlags, dsum, gsum, ssum)
end


"""
Convolution that allocates `P_conv` internally.
Don't use inside loops, use `mod!` instead.
"""
function conv!{T,Nd,Ng,Ns}(
	   d::AbstractArray{T,Nd}, 
	   g::AbstractArray{T,Ng},
	   s::AbstractArray{T,Ns}, attrib::Symbol)
	(attrib ∉ [:s, :d, :g]) && error("invalid attrib")
	dsize=[size(d)...]
	gsize=[size(g)...]
	ssize=[size(s)...]

	# allocation of freq matrices
	pa=P_conv(dsize=dsize, ssize=ssize, gsize=gsize, g=g, s=s, d=d)

	# using pa, return d, g, s according to attrib
	mod!(pa, attrib)
end

"""
Convolution modelling with no allocations at all.
By default, the fields `g`, `d` and `s` in pa are modified accordingly.
Otherwise use keyword arguments to input them.
"""
function mod!(pa::P_conv, attrib::Symbol; 
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

include("Pad.jl")
include("Xcorr.jl")
include("Misfits.jl")


end
