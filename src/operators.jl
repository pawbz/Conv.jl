
"""
A linear operator corresponding to convolution with either s or g
"""
function operator(pa, attrib)

	fw=(y,x)->F!(y, x, pa, attrib)
	bk=(y,x)->Fadj!(y, x, pa, attrib)

	return LinearMap{collect(typeof(pa).parameters)[1]}(fw, bk, 
		  length(pa.d),  # length of output
		  length(pa.g),  # length of output
		  ismutating=true)
end

function F!(y, x, pa, ::S)
	copyto!(pa.g,x)
	mod!(pa, D()) 
	copyto!(y,pa.d)
	return nothing
end

function Fadj!(y, x, pa, ::S)
	copyto!(pa.d,x)
	mod!(pa, G()) 
	copyto!(y,pa.g)
	return nothing
end

function F!(y, x, pa, ::G)
	copyto!(pa.s,x)
	mod!(pa, D()) 
	copyto!(y,pa.d)
	return nothing
end

function Fadj!(y, x, pa, ::G)
	copyto!(pa.d,x)
	mod!(pa, S()) 
	copyto!(y,pa.s)
	return nothing
end

function F!(y, x, pa, ::D)
	copyto!(pa.g,x)
	mod!(pa, S()) 
	copyto!(y,pa.s)
	return nothing
end

function Fadj!(y, x, pa, ::D)
	copyto!(pa.s,x)
	mod!(pa, G()) 
	copyto!(y,pa.g)
	return nothing
end
