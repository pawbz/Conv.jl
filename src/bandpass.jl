
mutable struct P_bandpass{T<:Real}
	dfreq::Array{Complex{T}}
	dfftp::FFTW.rFFTWPlan
	difftp::FFTW.Plan
	filter::Array{T}
end

function P_bandpass(T=Float64;
		    fmin, fmax, order, length)

	l=length
	nrfft=div(l,2)+1
	dfftp=plan_rfft(zeros(T, l),[1])
	difftp=plan_irfft(complex.(zeros(T, nrfft)),l,[1]) 

	dfreq=complex.(zeros(T,nrfft))
	filter=zeros(T,nrfft)

	filter=abs.(rfft(impz(b)))
end


function filt!(y,x,pa::P_bandpass)
	mul!(pa.dfreq, pa.dfftp, x)
	mul!(y, pa.difftp, pa.dfreq)
end
