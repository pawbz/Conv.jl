
"""
Type for correlated squared Euclidean
"""
mutable struct P_misfit_xcorr
	pxcorr::Conv.P_xcorr{Float64}
	cy::Vector{Matrix{Float64}}
	dcg::Vector{Matrix{Float64}}
end


function P_misfit_xcorr(nt::Int, nr::Int, 
			pxcorr=P_xcorr(nt, nr, norm_flag=false); y=nothing, cy=nothing)

	if(cy===nothing)
		cy=deepcopy(pxcorr.cg) # allocate cy
		(y===nothing) && error("need y or cy")
		(size(y)≠size(pxcorr.g)) && error("size y")

		Conv.mod!(pxcorr, cg=cy, g=y)
	end
	dcg=deepcopy(pxcorr.cg)
	fill!.(dcg, 0.0)
	return P_misfit_xcorr(pxcorr, cy, dcg)
end

function func_grad!(dfdx,  x,  pa::P_misfit_xcorr)
	nt=size(x,1)
	nr=size(x,2)
	cg=pa.pxcorr.cg
	cy=pa.cy
	dfdcg=pa.dcg

	# copy x
	copyto!(pa.pxcorr.g,x)

	# mod xcorr
	mod!(pa.pxcorr)

	J=0.0
	for ir in 1:length(cg)
		cgx=cg[ir]
		cyy=cy[ir]
		dfdcgx=dfdcg[ir]

		if(dfdx===nothing)
			JJ=Misfits.error_squared_euclidean!(nothing,  cgx,   cyy,   nothing, norm_flag=false)
		else
			JJ=Misfits.error_squared_euclidean!(dfdcgx,  cgx,   cyy,   nothing, norm_flag=false)
		end
		J+=JJ
	end
	if(!(dfdx===nothing))
		mod_grad!(dfdx, pa.pxcorr, dcg=dfdcg) 
	end

	return J
end

mutable struct P_misfit_weighted_acorr{T<:Real}
	p_conv::Conv.P_conv{T,2,2,2}
	ds::Matrix{Float64}
end


function P_misfit_weighted_acorr(nt::Int, nr::Int)
	p_conv=P_conv(gsize=[nt,nr], dsize=[nt,nr], ssize=[2*nt-1,nr], slags=[nt-1, nt-1])
	ds=zero(p_conv.s)
	return P_misfit_weighted_acorr(p_conv, ds)
end

"""
Each column of the matrix x is correlated with itself,
the energy at non-zero las will be penalized and returned as J.
"""
function func_grad!(dfdx, x, pa::P_misfit_weighted_acorr)
	nt=size(x,1)
	nr=size(x,2)

	copyto!(pa.p_conv.g,x)
	copyto!(pa.p_conv.d,x)

	Conv.mod!(pa.p_conv, S())
	s=pa.p_conv.s
	J=0.0
	for ir in 1:nr
		for it in 1:size(s,1)
			J += abs2(s[it,ir]) * abs2((nt-it)/(nt-1))
		end
	end

	if(!(dfdx===nothing))
		fill!(pa.ds, 0.0)
		for ir in 1:nr
			for it in 1:size(s,1)
				pa.ds[it,ir] = 2.0 * (s[it,ir]) * abs2((nt-it)/(nt-1))
			end
		end

		Conv.mod!(pa.p_conv, G(), g=dfdx, s=pa.ds)

		rmul!(dfdx, 2.)
	end

	return J

end

