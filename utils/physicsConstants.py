import math

PI = math.pi

# units
GHz     = pow(10.,9)
PHz     = pow(10.,15)
km      = 1.0e3
m       = 1.0
cm      = pow(10.,-2)
cm3     = pow(10.,-6)
cm2     = pow(10.,-4)
nm      = pow(10.,-9)
mum     = pow(10.,-6)
A       = pow(10.,-10)

eV      = 1.60217662*pow(10.,-19)
meV     = eV*1e-3
kV      = 1e3
mW      = 1.0e-3
J       = 1.0
MJ      = 1.0e6

s       = 1.0
kmh     = km/3600

kg      = 1.0
g       = 1.0e-3
grain   = 6.47989e-5
lbs     = 0.453592


# physics constants
q       = eV                        #C
kB      = 1.38064852*pow(10,-23)    #J/K
eps0    = 8.854187817*pow(10,-12)   #S.I.
mu0     = 4*PI*pow(10.,-7)          #H.m
hbar    = 1.0545718*pow(10,-34)     #J.s
h       = 2*PI*hbar;                #J.s
m0      = 9.1093856*pow(10,-31)     #kg
c       = 299792458                 #m/s

hbsm = pow(hbar/nm,2)/(2*m0)/eV # energy associated with 2pi nm wavelength
# alias 
Angstrum = pow(10.,-10)
c0       = 299792458                 


