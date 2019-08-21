
function Base.getindex(pa::P_conv, s::Symbol)
	@assert s in [:S, :G, :D]
	if(s==:S)
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
