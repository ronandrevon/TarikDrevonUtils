
#### gears
f_a = lambda z1,z2,m:(z1+z2)*m/2
f_d = lambda z,m:z*m
f_da = lambda z,m:z*m+2*m
z_in = lambda a,m,r:int(2*a/m/(1+r))
z_out = lambda r,zin:int(zin*r)
r = lambda zout,zin:zout/zin


def z1_from_a_z2(a,m,z2):
    z1 = int(2*a/m -z2)
    print('a = %.2f' %f_a(z1,z2,m) )
    return z1
