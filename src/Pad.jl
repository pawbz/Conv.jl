

"""
Methods to perform zero padding and truncation.

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
* `pad` means xpow2 is returned using x
* `truncate` means x is returned using xpow2
"""
function pad!{T}(
				  x::AbstractArray{T}, 
				  xpow2::AbstractArray{T}, 
				  nplags::Integer, 
				  nnlags::Integer, 
				  npow2::Integer, 
				  )
	(size(x,1) ≠ nplags + nnlags + 1) && error("size x")
	(size(xpow2,1) ≠ npow2) && error("size xpow2")

	for id in 1:size(x,2)
		xpow2[1,id] = (x[nnlags+1,id]) # zero lag
		# +ve lags
		if (nplags > 0) 
			for i=1:nplags
				@inbounds xpow2[i+1,id]= (x[nnlags+1+i,id])
			end
		end
		# -ve lags
		if(nnlags != 0) 
			for i=1:nnlags
				@inbounds xpow2[npow2-i+1,id] =(x[nnlags+1-i,id])
			end
		end
	end
	return nothing
end


function truncate!{T}(
				  x::AbstractArray{T}, 
				  xpow2::AbstractArray{T}, 
				  nplags::Integer, 
				  nnlags::Integer, 
				  npow2::Integer, 
				  )
	(size(x,1) ≠ nplags + nnlags + 1) && error("size x")
	(size(xpow2,1) ≠ npow2) && error("size xpow2")

	for id in 1:size(x,2)
		x[nnlags+1,id] = (xpow2[1,id]); # zero lag
		if(nplags != 0) 
			for i=1:nplags
				@inbounds x[nnlags+1+i,id] = (xpow2[1+i,id]);
			end
		end
		if(nnlags != 0)
			for i=1:nnlags
				@inbounds x[nnlags+1-i,id] = (xpow2[npow2-i+1,id])
			end
		end
	end
	return nothing
end


