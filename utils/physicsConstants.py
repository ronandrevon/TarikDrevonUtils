import math

############################# units #######################################
#m,s,kg
km      = 1e3
#m       = 1.0
cm      = pow(10.,-2)
cm3     = cm**3
cm2     = cm**2
mum     = 1e-6
nm      = 1e-9
A       = 1e-10
#s       = 1.0
kHz     = 1e3
MHz     = 1e6
GHz     = 1e9
THz     = 1e12
PHz     = 1e15
#kg      = 1.0
g       = 1e-3
##energy
#J       = 1.0
eV      = 1.60217662*pow(10.,-19)
keV     = eV*1e3
meV     = eV*1e6
meV     = eV*1e-3
kV      = 1e3
muW      = 1e-6
mW      = 1e-3
kW      = 1e3
MJ      = 1e6
## weird units
grain   = 6.47989e-5
lbs     = 0.453592
kmh     = km/3600

########################### physics constants ############################
q       = eV                        #C
kB      = 1.38064852*pow(10,-23)    #J/K
eps0    = 8.854187817*pow(10,-12)   #S.I.
mu0     = 4*math.pi*pow(10.,-7)          #H.m
hbar    = 1.0545718*pow(10,-34)     #J.s
h       = 2*math.pi*hbar;                #J.s
m0      = 9.1093856*pow(10,-31)     #kg
emass   = 510.99906                 #keV
c       = 299792458                 #m/s


# alias
Angstrum = pow(10.,-10)
c0       = 299792458

############### physics values ################################
alpha_cst = 1/137
a0 = hbar/(m0*c*alpha_cst)/A
mc2 = 510.99906 # keV
hbsm = pow(hbar/nm,2)/(2*m0)/eV # eV (k=1nm^-1)
