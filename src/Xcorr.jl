

"""
Plan to cross-correlate every possible pair of columns.
"""
mutable struct P_xcorr{T<:Real}
	gsize::Vector{Int}
	cgsize::Vector{Vector{Int}}
	cg::Vector{Matrix{T}}
	g::Array{T,2}
	paconv::P_conv{T,1,1,1}
	cg_indices::Vector{Int64}
	norm_flag::Bool
end

function P_xcorr(nt::Int64, nr::Int64; 
		 cg_indices::Vector{Int64}=collect(1:nr), 
		 cglags=[nt-1, nt-1], 
		 norm_flag=true)
	nts=sum(cglags)+1
	s=zeros(nts);
	g=zeros(nt);
	d=zeros(nt);

	# allocate g
	gsize=[nt, nr]
	gmat=zeros(gsize...) 

	# allocate cg
	cgsize=[[nts, nr-cg_indices[i]+1] for i in eachindex(cg_indices)]
	cg=[zeros(cgsize[i]...) for i in eachindex(cg_indices)]

	# create a plan for conv
	paconv=P_conv(gsize=[nt], dsize=[nt], ssize=[nts], g=g, s=s, d=d, slags=cglags)

	# create a plan for xcorr
	pa=P_xcorr(gsize, cgsize, cg, gmat, paconv, cg_indices, norm_flag)

	# return plan
	return pa
end

function xcorr(g::AbstractArray{Float64}, pa=P_xcorr(size(g,1), size(g,2))) 

	mod!(pa, g=g)

	return pa.cg 
end

"""
Use first colomn of g and cross-correlate with rest of columns of it.
And store results in cg.
After xcorr, the coeffcients are normalized with norm(g[:,1])
By default, cg has almost same positive and negative lags.
"""
function mod!(pa::P_xcorr; cg=pa.cg, g=pa.g)

	nr=size(g,2)
	cg_indices=pa.cg_indices
	lags=pa.paconv.slags
	norm_flag=pa.norm_flag

	any([(size(cg[ir]) ≠ (sum(lags)+1,nr-ir+1)) for ir in pa.cg_indices]) && error("size cg")

	irrr=0
	α=1.0
	nt=size(g,1)
	nts=sum(lags)+1
	ir1=0
	for ir in cg_indices
		ir1+=1
		cgx=cg[ir1]
		for i in 1:nt
			pa.paconv.d[i]=g[i,ir]
		end
		if(ir1==1 && norm_flag) # save scale reference for first column only 
			for i in 1:nt
				α+=g[i,ir]*g[i,ir]
			end
			α = (iszero(α)) ? 1.0 : inv(α) # take inverse if not zero
		end
		for (iir2,ir2) in enumerate(ir:nr)
			for i in eachindex(pa.paconv.g)
				pa.paconv.g[i]=g[i,ir2]
			end
			mod!(pa.paconv, :s)
		        for i in 1:nts
		        	cgx[i,iir2]=pa.paconv.s[i]
		        end
		        if(norm_flag) 
		        	for i in 1:nts
		        		cgx[i,iir2]*=α
		        	end
			end
		end
	end
	return pa
end

"""
given dJdcg and g, computes dJdg
"""
function mod_grad!(dg::AbstractArray{Float64}, pa::P_xcorr; dcg=pa.cg, g=pa.g)
	nr=size(g,2)
	nt=size(g,1)
	cg_indices=pa.cg_indices
	lags=pa.paconv.slags
	nts=sum(lags)+1
	for i in eachindex(dg)
		dg[i]=0.0
	end

	ir1=0
	for ir in cg_indices
		ir1+=1
		dcgx=dcg[ir1]

		for (iir2,ir2) in enumerate(ir:nr)

			for i in 1:nt
				pa.paconv.g[i]=g[i,ir2]
			end
			for i in 1:nts
				pa.paconv.s[i]=dcgx[i,iir2]
			end

			mod!(pa.paconv, :d)

			for i in 1:nt
				dg[i,ir]+=pa.paconv.d[i]
			end
		end

		for i in 1:nt
			pa.paconv.d[i]=g[i,ir]
		end

		for (iir2,ir2) in enumerate(ir:nr)
			for i in 1:nts
				pa.paconv.s[i]=dcgx[i,iir2]
			end

			mod!(pa.paconv, :g) 
			for i in 1:nt
				dg[i,ir2] +=pa.paconv.g[i]
			end
		end
	end
	return dg

end



"""
Convert Array{Array{Float64,2},1} to 
Array{Float64,2} and vice versa
"""
function cgmat!(cgmat, cg, flag; cg_indices::Vector{Int}=collect(1:length(cg)), 
					cgmat_indices::Vector{Int}=collect(1:length(cg)))
	nr=length(cg);
	nt=size(cgmat,1)

	temp=zeros(nt)
	if(flag==1)
		for (ir,ircg) in enumerate(cg_indices)
			nr1=nr-ir+1
			cgx=cg[ir]
			for irr in 1:nr1
				irm1=findin(cgmat_indices, ircg)[1]
				irm2=findin(cgmat_indices, cg_indices[ir+irr-1])[1]
				if(irm1<=irm2)
					igmat=sum([nr-i+1 for i in 1:irm1-1])+1+(irm2-irm1)
					for i in eachindex(temp)
						temp[i]=cgmat[i, igmat]
					end
				else
					igmat=sum([nr-i+1 for i in 1:irm2-1])+1+(irm1-irm2)
					for i in eachindex(temp)
						temp[i]=cgmat[nt-i+1, igmat]
					end
				end
				for i in eachindex(temp)
					cgx[i, irr]=temp[i]
				end
			end
		end
	end
	if(flag==-1)
		ntot=0
		for (ir,ircgmat) in enumerate(cgmat_indices)
			nr1=nr-ir+1
			for irr in 1:nr1
				irm1=findin(cg_indices, ircgmat)[1]
				irm2=findin(cg_indices, cgmat_indices[ir+irr-1])[1]
				if(irm1<=irm2)
					cgx=cg[irm1]
					for i in eachindex(temp)
						temp[i]=cgx[i,irm2-irm1+1]
					end
				else
					cgx=cg[irm2]
					for i in eachindex(temp)
						temp[i]=cgx[nt-i+1,irm1-irm2+1]
					end
				end
				for i in eachindex(temp)
					cgmat[i, ntot+irr]=temp[i]
				end
			end
			ntot+=nr1
		end
	end
end

function cgmat(cgmat::AbstractArray{Float64,2}, nr::Int; cg_indices=collect(1:nr))
	nts=size(cgmat,1)
	cg=[zeros(nts, nr-i+1) for i in 1:nr]
	cgmat!(cgmat, cg, 1, cg_indices=cg_indices)
	return cg
end

function cgmat(cg::Vector{Matrix{Float64}}, nr::Int;cgmat_indices=collect(1:nr))
	nts=size(cg[1],1)
	cgmat=zeros(nts,binomial(nr, 2)+nr)
	cgmat!(cgmat, cg, -1, cgmat_indices=cgmat_indices)
	return cgmat
end
