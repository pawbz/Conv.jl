using Conv
using Base.Test
using BenchmarkTools

np=1000
n=777
n2=100
x = randn(n,n2); xa = similar(x)
z = zeros(np,n2);

# cover all lags for serial mode
@time for i in [0, 2, 4, n-1]
	Conv.pad_truncate!(x, z, n-i-1,i,np,1)
	Conv.pad_truncate!(xa, z, n-i-1,i,np,-1)
	@test x ≈ xa
end
i=4
println("testing memory for pad truncates...")
@btime Conv.pad_truncate!(x, z, n-i-1,i,np,1)
@btime Conv.pad_truncate!(xa, z, n-i-1,i,np,-1)
println("...")


np=1000
n=777
x = randn(n); xa = similar(x)
z = zeros(np);
@time for i in [0, 2, 4, n-1]
	Conv.pad_truncate!(x, z, n-i-1,i,np,1)
	Conv.pad_truncate!(xa, z, n-i-1,i,np,-1)
	@test x ≈ xa
end
i=4
println("testing memory for pad truncates...")
@btime Conv.pad_truncate!(x, z, n-i-1,i,np,1)
@btime Conv.pad_truncate!(xa, z, n-i-1,i,np,-1)
println("...")



# fast_filt
# note that this test works for only some delta functions
for nw in [7, 8], nr in [20, 10], ns in [11, 10]

	r = zeros(nr); w = zeros(nw); s = zeros(ns);
	ra = similar(r); wa = similar(w); sa = similar(s);

	r[5]=1.; w[3]=1.;

	Conv.conv!(s,r,w,:d)
	Conv.conv!(s,r,wa,:s)
	@test w ≈ wa

	Conv.conv!(s,ra,w,:g)
	@test ra ≈ r

	Conv.conv!(sa,ra,wa,:d)
	@test sa ≈ s
end


## dot product test for fast filt

function filt_loop(func, n2; )
	nwvec=[101, 100]
	nrvec=[1000, 1001]
	nsvec=[1500, 901]
	np2=2024;
	for nw in nwvec, nr in nrvec, ns in nsvec

		r=randn(nr, n2...)
		s=randn(ns, n2...)
		w=randn(nw, n2...)
		func(s,r,w,:d)

		sa=randn(ns, n2...);
		ra=similar(r)
		func(sa,ra,w,:g)

		# dot product test
		@test dot(s, sa) ≈ dot(ra, r)

		r=randn(nr, n2...)
		s=randn(ns, n2...)
		w=zeros(nw, n2...)
		func(s,r,w,:s)


		wa=randn(nw, n2...);
		ra=similar(r);
		func(s,ra,wa,:g)

		# dot product test
		@test dot(r, ra) ≈ dot(wa, w)
	end
end

n2=128
@time filt_loop(Conv.conv!, n2)
@time filt_loop(Conv.conv!, n2)


using BenchmarkTools
# check if mod! will not have any allocations
n=10000
nr=1
println("====================")
d=randn(n,nr)
g=randn(n,nr)
s=randn(n,nr)
pa=Conv.P_conv(d=d,g=g,s=s, gsize=[n,nr], dsize=[n,nr], ssize=[n,nr]);

@btime Conv.mod!(pa, :d);
@btime Conv.mod!(pa, :d, d=d, g=g);
@btime Conv.mod!(pa, :g);
@btime Conv.mod!(pa, :s);

n=1000
nr=100
println("====================")
d=randn(n,nr)
g=randn(n,nr)
s=randn(n)
pa=Conv.P_conv(d=d,g=g,s=s, gsize=[n,nr], dsize=[n,nr], ssize=[n]);

@btime Conv.mod!(pa, :d);
@btime Conv.mod!(pa, :d, d=d, g=g);
@btime Conv.mod!(pa, :g);
@btime Conv.mod!(pa, :s);

println("====================")
d=randn(n,nr)
g=randn(n,nr)
s=randn(n,nr)
pa=Conv.P_conv(d=d,g=g,s=s, gsize=[n,nr], dsize=[n,nr], ssize=[n,nr]);

@btime Conv.mod!(pa, :d);
@btime Conv.mod!(pa, :d, d=d, g=g);
@btime Conv.mod!(pa, :g);
@btime Conv.mod!(pa, :s);




# =================================================
# weighted norm after auto-correlation
# =================================================
n1=100
n2=10

x=randn(n1,n2);
w=randn(n1,n2);
dfdx1=similar(x);
pa=Conv.P_misfit_weighted_acorr(n1,n2)

@time Conv.func_grad!(dfdx1,x,pa)
function func(x) 
	xx=reshape(x,n1,n2)
	return	Conv.func_grad!(nothing,xx,pa)
end
dfdx2=Calculus.gradient(func,vec(x));

@test dfdx1 ≈ reshape(dfdx2,n1,n2)


# =================================================
# squared euclidean after after xcorr
# =================================================
n1=50
n2=4

x=randn(n1,n2);
y=randn(n1,n2);
dfdx1=similar(x);
pa=Conv.P_misfit_xcorr(n1,n2, y=y)

@time Conv.func_grad!(dfdx1,x,pa)
function func(x) 
	xx=reshape(x,n1,n2)
	return	Conv.func_grad!(nothing,xx,pa)
end
dfdx2=Calculus.gradient(func,vec(x));

@test dfdx1 ≈ reshape(dfdx2,n1,n2)



