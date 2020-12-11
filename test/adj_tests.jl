n=100
nr=3
d=randn(n,nr)
g=randn(n,nr)
s=randn(n,nr)
pa=Conv.P_conv(d=d,g=g,s=s, gsize=[n,nr], dsize=[n,nr], ssize=[n,nr]);

function adjtest()
	x=randn(size(F,2))
	y=randn(size(F,1))
	a=LinearAlgebra.dot(y,F*x)
	b=LinearAlgebra.dot(x,adjoint(F)*y)
	c=LinearAlgebra.dot(x, transpose(F)*F*x)
	println("adjoint test: ", a, "\t", b)       
	@test isapprox(a,b,rtol=1e-6)
	println("must be positive: ", c)
	@test c>0.0
end

function do_tests(m=3)

	@testset "S" begin
		global pa
		global F=Conv.operator(pa, Conv.S());
		adjtest()
	end

	@testset "G" begin
		global pa
		global F=Conv.operator(pa, Conv.G());
		adjtest()
	end

	@testset "D" begin
		global pa
		global F=Conv.operator(pa, Conv.D());
		adjtest()
	end
end
do_tests()


n=100
nr=3
d=randn(n,nr)
g=randn(n,nr)
s=randn(n)
pa=Conv.P_conv(d=d,g=g,s=s, gsize=[n,nr], dsize=[n,nr], ssize=[n]);
@testset "S" begin
	global pa
	global F=Conv.operator(pa, Conv.S());
	adjtest()
end





#=
@testset "BD" begin
	global p=pa.plsbd
	for attrib in [FBD.S(), FBD.G()]
		global F=FBD.operator(p, attrib);
		adjtest()
	end
end



# adj tests of filter
pa=FBD.P_bandpass(Float64, fmin=0.1, fmax=0.4, nt=101)
global F=FBD.create_operator(pa)
@testset "bandpass" begin
	adjtest()
end
=#
