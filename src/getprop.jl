
"""
Return derived fields of `P_conv` 
* for `freq`, do zero padding, and then rFFT to return 
"""
function Base.getindex(pa::P_conv, s::Symbol)
	@assert s in [:sfreq, :dfreq, :gfreq, ]

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
	end
end


