
"""
Return derived fields of `P_conv` 
* for `freq`, do zero padding, and then rFFT to return 
"""
function Base.getindex(pa::P_conv, s::Symbol)
	@assert s in [:sfreq, :dfreq, :gfreq, :S,:G,:D]

	if(s==:sfreq)
		initialize_s!(pa)
		mul!(pa.sfreq, pa.sfftp, pa.spad)
		return pa.sfreq
	elseif(s==:gfreq)
		initialize_g!(pa)
		mul!(pa.gfreq, pa.gfftp, pa.gpad)
		return pa.gfreq
	elseif(s==:dfreq)
		initialize_d!(pa)
		mul!(pa.dfreq, pa.dfftp, pa.dpad)
		return pa.dfreq
	elseif(s==:S)
		return operator(pa, S())
	elseif(s==:G)
		return operator(pa, G())
	elseif(s==:D)
		return operator(pa, D())
	end
end

function S(s; args...)
	pa=P_conv(s=s; ssize=vcat(size(s)...), args...);
	return pa[:S]
end
