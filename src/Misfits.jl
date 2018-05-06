
"""
Type for correlated squared Euclidean
"""
mutable struct P_misfit_xcorr
	pxcorr::Conv.P_xcorr
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
	for i in eachindex(dcg)
		dcg[i][:]=0.0
	end
	return P_misfit_xcorr(pxcorr, cy, dcg)
end

function func_grad!(dfdx,  x,  pa::P_misfit_xcorr)
	nt=size(x,1)
	nr=size(x,2)
	cg=pa.pxcorr.cg
	cy=pa.cy
	dfdcg=pa.dcg

	# copy x
	for i in eachindex(pa.pxcorr.g)
		pa.pxcorr.g[i]=x[i]
	end

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


#=
"""
Each column of the matrix x is correlated with itself,
the energy at non-zero las will be penalized and returned as J.
"""
function error_acorr_weighted_norm!(dfdx, x; paconv=nothing, dfdwav=nothing)
	nt=size(x,1)
	nr=size(x,2)
	# create Conv mod if not preallocated
	if(paconv===nothing)
		paconv=Conv.Param(ntgf=nt, ntd=nt, ntwav=2*nt-1, dims=(nr,), wavlags=[nt-1, nt-1])
	end

	copy!(paconv.gf,x)
	copy!(paconv.d,x)

	Conv.mod!(paconv, :wav)
	wav=paconv.wav
	J=0.0
	for ir in 1:nr
		for it in 1:size(wav,1)
			J += (wav[it,ir]) * (wav[it,ir]) * abs((nt-it)/(nt-1))
		end
	end

	if(!(dfdx===nothing))

		if(dfdwav===nothing)
			dfdwav=zeros(paconv.wav)
		end
		for ir in 1:nr
			for it in 1:size(wav,1)
				dfdwav[it,ir] = 2.0 * (wav[it,ir]) * abs((nt-it)/(nt-1))
			end
		end

		Conv.mod!(paconv, :gf, gf=dfdx, wav=dfdwav)

		scale!(dfdx, 2.)
	end

	return J

end


=#
