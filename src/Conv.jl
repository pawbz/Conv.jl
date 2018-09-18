"""
This module is used in DeConv, ConvMix and Coupling
"""
module Conv
using FFTW
using LinearAlgebra

using DSP: nextfastfft
using Misfits

struct D end
struct G end
struct S end

"""
Model : d = convolution(g, s)
d, g and s can have arbitary +ve and -ve lags
"""
mutable struct P_conv{T<:Real, Nd, Ng, Ns}
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
end

function Base.fill!(pa::P_conv, k)
	for iff in [:d, :g, :s]
		m=getfield(pa,iff)
		for i in eachindex(m)
			m[i]=eltype(pa.d)(k)
		end
	end
end



function P_conv(T=Float64; 
	       dsize::Vector{Int64}=[1],
	       ssize::Vector{Int64}=[1],
	       gsize::Vector{Int64}=[1],
	       d=zeros(T,dsize...), 
	       s=zeros(T,ssize...),
	       g=zeros(T,gsize...),
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

	FFTW.set_num_threads(Sys.CPU_THREADS)
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

	pa=P_conv(dsize, gsize, ssize, np2, d, g, s, 
	  dpad, gpad, spad,
	  dfreq, gfreq, sfreq, 
	  dfftp, difftp, 
	  gfftp, gifftp, 
	  sfftp, sifftp, 
	  glags, slags, dlags)
end


"""
Convolution that allocates `P_conv` internally.
Don't use inside loops, use `mod!` instead.
"""
function conv!(
	   d::AbstractArray{T,Nd}, 
	   g::AbstractArray{T,Ng},
	   s::AbstractArray{T,Ns}, attrib::Union{D,G,S}) where {T,Nd,Ng,Ns}
	dsize=[size(d)...]
	gsize=[size(g)...]
	ssize=[size(s)...]

	# allocation of freq matrices
	pa=P_conv(dsize=dsize, ssize=ssize, gsize=gsize, g=g, s=s, d=d)

	# using pa, return d, g, s according to attrib
	mod!(pa, attrib)
end

function initialize_d!(pa::P_conv, d=pa.d)
	T=eltype(pa.d)
	fill!(pa.dfreq,complex(T(0)))
	fill!(pa.dpad,T(0))
	pad!(d, pa.dpad, pa.dlags[1], pa.dlags[2], pa.np2)
end

function initialize_g!(pa::P_conv, g=pa.g)
	T=eltype(pa.g)
	fill!(pa.gfreq, complex(T(0)))
	fill!(pa.gpad,T(0))
	pad!(g, pa.gpad, pa.glags[1], pa.glags[2], pa.np2)
end


function initialize_s!(pa::P_conv, s=pa.s)
	T=eltype(pa.s)
	fill!(pa.sfreq, complex(T(0)))
	fill!(pa.spad, T(0))
	pad!(s, pa.spad, pa.slags[1], pa.slags[2], pa.np2)
end

# initialize freq vectors
function initialize_all!(pa::P_conv, d=pa.d, g=pa.g, s=pa.s)
	initialize_d!(pa, d);	initialize_g!(pa, g);    initialize_s!(pa, s)
end

"""
Convolution modelling with no allocations at all.
By default, the fields `g`, `d` and `s` in pa are modified accordingly.
Otherwise use keyword arguments to input them.
"""
function mod!(pa::P_conv{T,N,N,N}, ::D; g=pa.g, d=pa.d, s=pa.s) where {N,T<:Real}
	initialize_all!(pa, d, g, s)
	mul!(pa.sfreq, pa.sfftp, pa.spad)
	mul!(pa.gfreq, pa.gfftp, pa.gpad)
	for i in eachindex(pa.dfreq)
		@inbounds pa.dfreq[i] = pa.gfreq[i] * pa.sfreq[i]
	end
	mul!(pa.dpad, pa.difftp, pa.dfreq)
	truncate!(d, pa.dpad, pa.dlags[1], pa.dlags[2], pa.np2)
	return pa
end

function mod!(pa::P_conv{T,N,N,N}, ::G; g=pa.g, d=pa.d, s=pa.s) where {N,T<:Real}
	initialize_all!(pa, d, g, s)
	mul!(pa.sfreq, pa.sfftp, pa.spad)
	mul!(pa.dfreq, pa.dfftp, pa.dpad)
	conj!(pa.sfreq)
	for i in eachindex(pa.dfreq)
		@inbounds pa.gfreq[i] = pa.dfreq[i] * pa.sfreq[i]
	end
	mul!(pa.gpad, pa.gifftp, pa.gfreq)
	truncate!(g, pa.gpad, pa.glags[1], pa.glags[2], pa.np2)
	return pa
end
		
function mod!(pa::P_conv{T,N,N,N}, ::S; g=pa.g, d=pa.d, s=pa.s) where {N,T<:Real}
	initialize_all!(pa, d, g, s)
	mul!(pa.gfreq, pa.gfftp, pa.gpad)
	mul!(pa.dfreq, pa.dfftp, pa.dpad)
	conj!(pa.gfreq)
	for i in eachindex(pa.dfreq)
		@inbounds pa.sfreq[i] = pa.dfreq[i] * pa.gfreq[i]
	end
	mul!(pa.spad, pa.sifftp, pa.sfreq)
	truncate!(s, pa.spad, pa.slags[1], pa.slags[2], pa.np2)
	return pa
end



# dsum=true
function mod!(pa::P_conv{T,1,2,2}, ::D; g=pa.g, d=pa.d, s=pa.s) where {T<:Real}
	initialize_all!(pa, d, g, s)
	mul!(pa.sfreq, pa.sfftp, pa.spad)
	mul!(pa.gfreq, pa.gfftp, pa.gpad)
	for i in CartesianIndices(size(pa.gfreq))
		@inbounds pa.dfreq[i[1]] += pa.gfreq[i] * pa.sfreq[i]
	end
	mul!(pa.dpad, pa.difftp, pa.dfreq)
	truncate!(d, pa.dpad, pa.dlags[1], pa.dlags[2], pa.np2)
	return pa
end

function mod!(pa::P_conv{T,1,2,2}, ::G; g=pa.g, d=pa.d, s=pa.s) where {T<:Real}
	initialize_all!(pa, d, g, s)
	mul!(pa.sfreq, pa.sfftp, pa.spad)
	mul!(pa.dfreq, pa.dfftp, pa.dpad)
	conj!(pa.sfreq)
	for i in CartesianIndices(size(pa.gfreq))
		@inbounds pa.gfreq[i] = pa.dfreq[i[1]] * pa.sfreq[i]
	end
	mul!(pa.gpad, pa.gifftp, pa.gfreq)
	truncate!(g, pa.gpad, pa.glags[1], pa.glags[2], pa.np2)
	return pa
end
		
function mod!(pa::P_conv{T,1,2,2}, ::S; g=pa.g, d=pa.d, s=pa.s) where {T<:Real}
	initialize_all!(pa, d, g, s)
	mul!(pa.gfreq, pa.gfftp, pa.gpad)
	mul!(pa.dfreq, pa.dfftp, pa.dpad)
	conj!(pa.gfreq)
	for i in CartesianIndices(size(pa.gfreq))
		@inbounds pa.sfreq[i] = pa.dfreq[i[1]] * pa.gfreq[i]
	end
	mul!(pa.spad, pa.sifftp, pa.sfreq)
	truncate!(s, pa.spad, pa.slags[1], pa.slags[2], pa.np2)
	return pa
end

# gsum=true
function mod!(pa::P_conv{T,2,1,2}, ::D;  g=pa.g, d=pa.d, s=pa.s) where {T<:Real}
	initialize_all!(pa, d, g, s)
	mul!(pa.sfreq, pa.sfftp, pa.spad)
	mul!(pa.gfreq, pa.gfftp, pa.gpad)
	for i in CartesianIndices(size(pa.sfreq))
		@inbounds pa.dfreq[i] = pa.gfreq[i[1]] * pa.sfreq[i]
	end
	mul!(pa.dpad, pa.difftp, pa.dfreq)
	truncate!(d, pa.dpad, pa.dlags[1], pa.dlags[2], pa.np2)
	return pa
end
function mod!(pa::P_conv{T,2,1,2}, ::G;  g=pa.g, d=pa.d, s=pa.s) where {T<:Real}
	initialize_all!(pa, d, g, s)
	mul!(pa.sfreq, pa.sfftp, pa.spad)
	mul!(pa.dfreq, pa.dfftp, pa.dpad)
	conj!(pa.sfreq)
	for i in CartesianIndices(size(pa.sfreq))
		@inbounds pa.gfreq[i[1]] += pa.dfreq[i] * pa.sfreq[i]
	end
	mul!(pa.gpad, pa.gifftp, pa.gfreq)
	truncate!(g, pa.gpad, pa.glags[1], pa.glags[2], pa.np2)
	return pa
end
		
function mod!(pa::P_conv{T,2,1,2}, ::S;  g=pa.g, d=pa.d, s=pa.s) where {T<:Real}
	initialize_all!(pa, d, g, s)
	mul!(pa.gfreq, pa.gfftp, pa.gpad)
	mul!(pa.dfreq, pa.dfftp, pa.dpad)
	conj!(pa.gfreq)
	for i in CartesianIndices(size(pa.sfreq))
		@inbounds pa.sfreq[i] = pa.dfreq[i] * pa.gfreq[i[1]]
	end
	mul!(pa.spad, pa.sifftp, pa.sfreq)
	truncate!(s, pa.spad, pa.slags[1], pa.slags[2], pa.np2)
	return pa
end

# ssum=true
function mod!(pa::P_conv{T,2,2,1}, ::D; g=pa.g, d=pa.d, s=pa.s) where {T<:Real}
	initialize_all!(pa, d, g, s)
	mul!(pa.sfreq, pa.sfftp, pa.spad)
	mul!(pa.gfreq, pa.gfftp, pa.gpad)
	for i in CartesianIndices(size(pa.gfreq))
		@inbounds pa.dfreq[i] = pa.gfreq[i] * pa.sfreq[i[1]]
	end
	mul!(pa.dpad, pa.difftp, pa.dfreq)
	truncate!(d, pa.dpad, pa.dlags[1], pa.dlags[2], pa.np2)
	return pa
end
function mod!(pa::P_conv{T,2,2,1}, ::G; g=pa.g, d=pa.d, s=pa.s) where {T<:Real}
	initialize_all!(pa, d, g, s)
	mul!(pa.sfreq, pa.sfftp, pa.spad)
	mul!(pa.dfreq, pa.dfftp, pa.dpad)
	conj!(pa.sfreq)
	for i in CartesianIndices(size(pa.gfreq))
		@inbounds pa.gfreq[i] = pa.dfreq[i] * pa.sfreq[i[1]]
	end
	mul!(pa.gpad, pa.gifftp, pa.gfreq)
	truncate!(g, pa.gpad, pa.glags[1], pa.glags[2], pa.np2)
	return pa
end
function mod!(pa::P_conv{T,2,2,1}, ::S; g=pa.g, d=pa.d, s=pa.s) where {T<:Real}
	initialize_all!(pa, d, g, s)
	mul!(pa.gfreq, pa.gfftp, pa.gpad)
	mul!(pa.dfreq, pa.dfftp, pa.dpad)
	conj!(pa.gfreq)
	for i in CartesianIndices(size(pa.gfreq))
		@inbounds pa.sfreq[i[1]] += pa.dfreq[i] * pa.gfreq[i]
	end
	mul!(pa.spad, pa.sifftp, pa.sfreq)
	truncate!(s, pa.spad, pa.slags[1], pa.slags[2], pa.np2)
	return pa
end






include("Pad.jl")
include("Xcorr.jl")
include("Misfits.jl")


end
