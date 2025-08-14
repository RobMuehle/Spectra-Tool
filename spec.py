import math
import copy
import argparse
import numpy as np

from numpy.linalg import norm
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse import linalg


exec_spectrum  = True
exec_read      = False
exec_norm      = False
exec_max       = False
exec_invert    = False
exec_corr      = False
exec_lin       = False
exec_lin_range = False
exec_baseline  = False


parser = argparse.ArgumentParser()
fgroup = parser.add_mutually_exclusive_group( required=True )
fgroup.add_argument( '--file',       dest="file" )
fgroup.add_argument( '--readspec',   dest="readspec" )
fgroup.add_argument( "--corr",       nargs=3, dest="corr" )
parser.add_argument( '--readres',    dest="readres" )
parser.add_argument( "--type",       dest="convolution", choices=[ "lorentz", "gauss", "pseudo-voigt" ], default="lorentz" )
parser.add_argument( "--fwhm",       dest="fwhm", default=float(20) )
parser.add_argument( "--parameter",  dest="parameter", default="none" )
parser.add_argument( "--norm",       nargs="?", dest="norm", choices=[ "l1", "l2", "inf-max" ], const="l2" )
parser.add_argument( "--width",      dest="width", type=float, default=float(2000) )
parser.add_argument( "--resolution", dest="resolution", type=float, default=float(1) )
parser.add_argument( "--offset",     dest="offset", default="0" )
parser.add_argument( "--invert",     action="store_true"  )
parser.add_argument( "--maxima",     action="store_true"  )
parser.add_argument( "--scale",      dest="scale", default=float(1.0)  )
parser.add_argument( "--lin",        dest="lin", action="store_true" )
parser.add_argument( "--lin_range",  dest="lin_range", action="store_true" )
parser.add_argument( '--baseline',   dest="blc", choices=[ "als", "arpls" ] )

args = parser.parse_args()

if args.file:
   input_file=args.file

if args.readspec:
   if not args.norm and not args.lin and not args.lin_range:
      print("\n Error: Option --readspec also requires option --norm or --lin\n"); quit()
   exec_read=True
   exec_spectrum=False
   input_file=args.readspec

if args.readres:
   res_file=args.readres

if args.corr:
   exec_corr=True
   exec_spectrum=False
   methods = [ "cauchy-schwarz", "euclidian", "pearson", "spearman" ]
   check=False
   for method in methods:
      if method in args.corr:
         args.corr.remove(method)
         corr_method=method
         check=True; break
   if not check:
      print("\n Error: Unknown correlation method requested\n"); quit()

if args.convolution:
   convolution=args.convolution

if args.fwhm:
   fwhm=float(args.fwhm)

if args.parameter:
   parameter_file=args.parameter

if args.offset:
   offset=int(args.offset)

if args.width:
   width=float(args.width)
   spectrum_range=( float(args.width) - float(offset) )

if args.resolution:
   resolution=float(args.resolution)

if args.norm:
   exec_norm=True
   norm=args.norm

if args.lin or args.lin_range:
   if not args.readspec:
      print("\n Error: Option --lin also requires option --readspec\n"); quit()
   lin_res=resolution
   exec_lin=True
   if args.lin_range:
      exec_lin_range=True
   exec_max=False

if args.invert:
   exec_invert=True
   exec_spectrum=False
   exec_norm=False
   exec_max=False

if args.maxima:
   exec_max=True

if args.scale:
   scale=float(args.scale)

if args.blc:
   exec_baseline=True
   baseline_type=args.blc
   exec_spectrum=False
   exec_norm=False
   exec_max=False



#################################################### FUNCTIONS ##############################################################
   

#============================================================================================================================
def LORENTZ( spectrum, frequency, intensity, fwhm ):
   Pi = math.pi
   gamma = fwhm/2
   for datapoint in range( len(spectrum) ):
      x = spectrum[datapoint][0]
      x_intense = intensity * ( (1.0/Pi)*(gamma/((gamma)**2 + (x-frequency)**2)) )
      spectrum[datapoint][1] += x_intense

   return None
#============================================================================================================================


#============================================================================================================================
def GAUSS( spectrum, frequency, intensity, fwhm ):
   Pi = math.pi
   sigma = fwhm/( 2*math.sqrt( 2*math.log(2) ) )
   print(sigma)
   for datapoint in range( len(spectrum) ):
      x = spectrum[datapoint][0]
      x_intense = ( intensity*( 1/(sigma*math.sqrt(2*Pi)) ) * math.exp( (-1.0/2.0) * ( ((x - frequency)/sigma)**2 ) ) )
      spectrum[datapoint][1] += x_intense

   return None
#============================================================================================================================


#============================================================================================================================
def PSEUDO_VOIGT( spectrum, frequency, intensity, fwhm ):
   Pi = math.pi
   spec_l = copy.deepcopy(spectrum)
   spec_g = copy.deepcopy(spectrum)
   spec_lg = []
   for datapoint in range( len(spectrum) ):
      spec_lg.append( [float(spectrum[datapoint][0]), float(0.0) ] )
   FWHM = ( fwhm**5 + 2.69269*(fwhm**5) + 2.42843*(fwhm**5) + 4.47163*(fwhm**5) + 0.07842*(fwhm**5) + fwhm**5)**(1.0/5.0)
   eta = ( 1.36603*(fwhm/FWHM) - 0.47719*(fwhm/FWHM)**2 + 0.11116*(fwhm/FWHM)**3 )
   LORENTZ( spec_l, frequency, intensity, fwhm )
   GAUSS( spec_g, frequency, intensity, fwhm )

   for datapoint in range( len(spec_lg) ):
      spec_lg[datapoint][1] +=  ( (1 - eta) * spec_g[datapoint][1] ) + ( eta * spec_l[datapoint][1] )

   return spec_lg
#============================================================================================================================


# Subtract global minimum value
#============================================================================================================================
def SUB_MIN( spectrum ):

   correction=min( spectrum )
   sub_int_vec=[]
   for index in range( 0, len(spectrum) ):
      sub_int_vec.append( spectrum[index]-correction )

   return sub_int_vec
#============================================================================================================================

 

#============================================================================================================================
def NORMALIZATION( spectrum, norm="l2" ):

   norm_spectrum = []
   for i in range ( len(spectrum) ):
      norm_spectrum.append( [ float(0.0), float(0.0) ] )

   x_i = 0.0

   if norm == "l1":
      for datapoint in range ( len(spectrum) ):
         x_i = x_i + abs( spectrum[datapoint][1] )
      norm_weight = x_i

   if norm == "l2":
      for datapoint in range ( len(spectrum) ):
         x_i = x_i + ( spectrum[datapoint][1] )**2
      norm_weight = math.sqrt(x_i)

   if norm == "inf-max":
      intensities = []
      for array in spectrum:
            intensities.append( float(array[1]) )
      x_i = max( intensities )
      norm_weight = x_i

   for datapoint in range ( len(spectrum) ):
      norm_spectrum[datapoint][0] = spectrum[datapoint][0]
      norm_spectrum[datapoint][1] = spectrum[datapoint][1]/norm_weight

   return norm_spectrum
#============================================================================================================================


#============================================================================================================================
def FIND_MAXIMA( spectrum ):
   maxima = {}
   downward_spiral = False
   for  singlepoint in range ( 0, spectrum_datapoints ):
      if singlepoint == 0:
         continue
      if spectrum[singlepoint][1] > spectrum[singlepoint-1][1]:
         downward_spiral = False
         continue
      if spectrum[singlepoint][1] < spectrum[singlepoint-1][1] and downward_spiral:
         continue
      else:
         maxima.update( { (offset+(singlepoint-1)*resolution) : spectrum[singlepoint-1][1] } )
         downward_spiral = True

   return maxima
#============================================================================================================================


#============================================================================================================================
def LIN_INT( data, res ):
   interpolation = []
   for dp in range( len(data) ):
      if dp == 0:
         continue
      else:
         if not (data[dp-1][0]).is_integer():
            if round(data[dp-1][0], 0) < data[dp-1][0]:
               x_start=round(data[dp-1][0]+0.499999, 0)
            else:
               x_start=round(data[dp-1][0], 0)
         else:
            x_start=data[dp-1][0]
         if not (data[dp][0]).is_integer():
            if round(data[dp][0], 0) > data[dp][0]:
               x_end=round(data[dp][0]-0.499999, 0)
            else:
               x_end=round(data[dp][0], 0)
         else:
           x_end=data[dp][0]-1
      n=int( (x_end - x_start) / res )+1
      for i in range( 0, n ):
         x_n = x_start+i*res
         y_n = ( data[dp-1][1] * ( data[dp][0] - x_n ) + data[dp][1] * ( x_n - data[dp-1][0] ) ) / ( data[dp][0] - data[dp-1][0] )
         interpolation.append( [x_n, y_n] )

   return interpolation
#============================================================================================================================


#============================================================================================================================
def INVERT( input_file ):
   data = []
   file=open( input_file, "r" )
   for line in file:
      line = line.strip()
      line = line.split()
      data.append( line )
   file.close()
   data.reverse()
   file=open( input_file+".inv", "w" )
   for vector in data:
      file.write( "{0} {1}\n".format( vector[0], vector[1] ) )
   file.close()
#============================================================================================================================


#============================================================================================================================
def CAUCHY_SCHWARZ( vector_1, vector_2 ):

   sum_vec_1_2_cross_product = 0.0
   sum_sqrd_vec_1_components = 0.0
   sum_sqrd_vec_2_components = 0.0
   for index in range( len(vector_1) ):
      sum_vec_1_2_cross_product = sum_vec_1_2_cross_product + ( vector_1[index][1] * vector_2[index][1] )
      sum_sqrd_vec_1_components = sum_sqrd_vec_1_components + ( vector_1[index][1]**2 )
      sum_sqrd_vec_2_components = sum_sqrd_vec_2_components + ( vector_2[index][1]**2 )

   return ( ( sum_vec_1_2_cross_product**2 )/( sum_sqrd_vec_1_components * sum_sqrd_vec_2_components ) )


def EUCLIDIAN_NORM( vector_1, vector_2 ):

   sum_sqrd_vec_1_2_diff = 0.0
   sum_sqrd_vec_2 = 0.0
   for index in range( len(vector_1) ):
      sum_sqrd_vec_1_2_diff = sum_sqrd_vec_1_2_diff + ( (vector_1[index][1] - vector_2[index][1])**2 )
      sum_sqrd_vec_2 = sum_sqrd_vec_2 + ( (vector_2[index][1])**2 )

   return ( 1/(1.0 + (sum_sqrd_vec_1_2_diff/sum_sqrd_vec_2) ) )


def PEARSON( vector_1, vector_2 ):

   sum_vector_1=0.0
   sum_vector_2=0.0
   for component in range( len(vector_1) ):
      sum_vector_1 = sum_vector_1 + (vector_1[component][1])
      sum_vector_2 = sum_vector_2 + (vector_2[component][1])
   vector_1_norm = sum_vector_1/(len(vector_1))
   vector_2_norm = sum_vector_2/(len(vector_2))

   p=0.0
   q=0.0
   for i in range( len(vector_1) ):
      p = p + ( ( vector_1[i][1] - vector_1_norm ) * ( vector_2[i][1] - vector_2_norm ) )
      q = q + ( math.sqrt(( vector_1[i][1] - vector_1_norm )**2) * math.sqrt(( vector_2[i][1] - vector_2_norm )**2 ) )
   r = p/q

   return r


def SPEARMAN( vector_1, vector_2 ):

   ranking = []
   for index in range( len(vector_1) ):
      ranking.append( [ float(vector_1[index][1]), float(vector_2[index][1]), 0, 0 ] )

   ranking.sort( key = lambda x: x[0] )
   rank=1
   for index in range( len(ranking) ):
      ranking[index][2] = rank
      rank+=1

   ranking.sort( key = lambda x: x[1] )
   rank=1
   for index in range( len(ranking) ):
      ranking[index][3] = rank
      rank+=1

   sum_di_2 = 0.0
   for index in range( len(ranking) ):
      sum_di_2 = sum_di_2 + ( ( ranking[index][3] - ranking[index][2] )**2 )
   n=len(ranking)

   return ( 1.0 - ( (6.0*sum_di_2)/(n*(n**2-1)) ) )
#============================================================================================================================


#============================================================================================================================
def BASELINE_ALS(y, lam=1.0e+4, p=1.0e-3, niter=10):

   L = len(y)
   D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
   w = np.ones(L)
   for i in range(niter):
      W = sparse.spdiags(w, 0, L, L)
      Z = W + lam * D.dot(D.transpose())
      z = spsolve(Z, w*y)
      w = p * (y > z) + (1-p) * (y < z)

   return z
#============================================================================================================================


#============================================================================================================================
def BASELINE_ARPLS(y, lam=1.0e+4, ratio=1.0e-6, niter=10, full_output=False):

    L = len(y)

    diag = np.ones(L - 2)
    D = sparse.spdiags([diag, -2*diag, diag], [0, -1, -2], L, L - 2)

    H = lam * D.dot(D.T)

    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)

    crit = 1
    count = 0

    while crit > ratio:
        z = linalg.spsolve(W + H, W * y)
        d = y - z
        dn = d[d < 0]

        m = np.mean(dn)
        s = np.std(dn)

        w_new = 1 / (1 + np.exp(2 * (d - (2*s - m))/s))

        crit = norm(w_new - w) / norm(w)

        w = w_new
        W.setdiag(w)

        count += 1

        if count > niter:
#            print('Maximum number of iterations exceeded')
            break

    if full_output:
        info = {'num_iter': count, 'stop_criterion': crit}
        return z, d, info
    else:
        return z
#============================================================================================================================



###################################################### MAIN #################################################################


# Prepare nested list of desired spectral range
#----------------------------------------------------------------------------------------------------------------------------
spectrum = []

if exec_read:
   try:
      with open(input_file, "r") as file:
         for line in file:
            line=line.strip()
            line=line.split()
#            spectrum.append( [ float(line[0])*scale, float(line[1]) ] )
            spectrum.append( [ float(line[0])*scale, float(line[1]) ] )
   except FileNotFoundError:
      print("\n Error: File '"+input_file+"' not found\n"); quit()
   for vector in spectrum:
      for index in range( len(vector) ):
         vector[index] = float( vector[index] )

elif args.readres:
   resolution = "custom"
   custom_resolution=[]
   try:
      with open( res_file, "r" ) as file:
         for line in file:
            line=line.strip()
            line=line.split()
            custom_resolution.append( float(line[0]) )
   except FileNotFoundError:
      print("\n Error: File '"+res_file+"' not found\n"); quit()
   for i in range ( len(custom_resolution) ):
      spectrum.append( [custom_resolution[i], float(0.0)] )

else:
   x = offset
   while x <= spectrum_range:
      spectrum.append( [float(x), float(0.0)] )
      x = x + resolution
#----------------------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------------------
if exec_spectrum:

   raw_spectrum = []
   try:
      with open( input_file, "r" ) as file:
         for line in file:
            line=line.strip()
            line=line.split()
            raw_spectrum.append( [ float(line[0])*scale, float(line[1]) ] )
         raw_spectrum_datapoints=len( raw_spectrum )
   except FileNotFoundError:
      print( "\n Error: File '"+input_file+"' not found\n" ); quit()
   for vector in raw_spectrum:
      for index in range( len(vector) ):
         vector[index] = float( vector[index] )

   print("\n Freq+Ints file : "+input_file)
   print(" Parameter file : "+parameter_file)
   print(" Total range    : {0:<7}{1:5}".format( spectrum_range, "cm^-1" ) )
   if type(resolution) == str:
      print(" Resolution     : {0:}".format( resolution ) )
   else:
      print(" Resolution     : {0:<7.1f}{1:5}".format( float(resolution), "cm^-1" ) )
   print(" Offset         : {0:<7.1f}{1:5}".format( float(offset), "cm^-1" ) )
   print(" Convolution    : {0:5}".format( convolution ) )
   print(" Fwhm           : {0:<7.1f}{1:5}".format( float(fwhm), "cm^-1" ) )
   if not exec_norm: print("")

   for frequency in range( len(raw_spectrum) ):
      raw_frequency = raw_spectrum[frequency][0]
      raw_intensity = raw_spectrum[frequency][1]

      if convolution == "lorentz":
         LORENTZ( spectrum, raw_frequency, raw_intensity, fwhm )
      elif convolution == "gauss":
         GAUSS( spectrum, raw_frequency, raw_intensity, fwhm )
      elif convolution == "pseudo-voigt":
         spectrum = PSEUDO_VOIGT( spectrum, raw_frequency, raw_intensity, fwhm )

   with open( input_file+".spec", "w" ) as file:
      for datapoint in range( len(spectrum) ):
         file.write( "{res:10.3f}{intense:15.6f}\n".format( res=(spectrum[datapoint][0]), intense=spectrum[datapoint][1] ) )
#----------------------------------------------------------------------------------------------------------------------------


# Normalize spectrum using L1-, L2-, or Inf-max norm
#----------------------------------------------------------------------------------------------------------------------------
if exec_norm:

   normalization = { "l1" : "L1 norm", "l2" : "L2 norm", "inf-max" : "L-infinity norm" } 
   print("\n Using "+normalization[norm]+" for data normalization" )
   norm_spectrum = NORMALIZATION( spectrum, norm )

   total_norm=0.0
   for datapoint in range( len(spectrum) ):
      if norm == "l1":
         total_norm = total_norm + (norm_spectrum[datapoint][1])
      elif norm == "l2":
         total_norm = total_norm + (norm_spectrum[datapoint][1])**2
      elif norm == "inf-max":
         intensities = []
         for array in norm_spectrum:
            intensities.append( float(array[1]) )
         total_norm = max( intensities )

   with open(input_file+".norm", "w") as file:
      for datapoint in range( len(spectrum) ):
         file.write( "{res:8.1f}{intense:13.5f}\n".format( res=(norm_spectrum[datapoint][0]), intense=norm_spectrum[datapoint][1] ) )

   print( " Total after normalization: {0:5.2f}\n".format( total_norm ) )
#----------------------------------------------------------------------------------------------------------------------------


# Calculate Cauchy-Schwarz/Euclidian/Pearson/Spearman correlation coefficient
#----------------------------------------------------------------------------------------------------------------------------
if exec_corr:

   print("\n Int. vec. 1 from file : "+args.corr[0])
   print(" Int. vec. 2 from file : "+args.corr[1])

   spectra_list = [ [], [] ]
   for spec_file in args.corr:
      try:
         with open( spec_file, "r" ) as file:
            for line in file:
               line=line.strip()
               line=line.split()
               spectra_list[args.corr.index(spec_file)].append( line )
      except FileNotFoundError:
         print("\n Error: File(s) '"+args.corr[0]+"' and/or '"+args.corr[0]+"' not found\n"); quit()
      for vector in spectra_list[args.corr.index(spec_file)]:
         for index in range( len(vector) ):
            vector[index] = float(vector[index])

   if len(spectra_list[0]) != len(spectra_list[1]):
      print("\n Error: intensity vector lengths do not match\n"); quit()

   correlation = {
      "cauchy-schwarz" : [ "Cauchy-Schwarz match",  CAUCHY_SCHWARZ( spectra_list[0], spectra_list[1] ) ],
      "euclidian"      : [ "Euclidian norm",        EUCLIDIAN_NORM( spectra_list[0], spectra_list[1] ) ],
      "pearson"        : [ "Pearson corr. coeff.",  PEARSON( spectra_list[0], spectra_list[1] ) ],
      "spearman"       : [ "Spearman corr. coeff.", SPEARMAN( spectra_list[0], spectra_list[1] ) ]
   }

   print( "\n {:22s}: {:0.4f}\n".format( correlation[corr_method][0], correlation[corr_method][1]) )
#----------------------------------------------------------------------------------------------------------------------------


# Print out spectrum peak maxima (simple algorithm)
#----------------------------------------------------------------------------------------------------------------------------
if exec_max:
   spectrum_maxima=FIND_MAXIMA( spectrum )
   file=open(input_file+".max", "w")
   for smax in sorted(spectrum_maxima.keys()):
      file.write( "{freq:9.2f}{intense:13.5f}\n".format( freq=smax, intense=spectrum_maxima[smax] ) )
   file.close()
#----------------------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------------------
if exec_lin:
   lin_spectrum=LIN_INT( spectrum, lin_res )

   with open( input_file+".lin.spec", "w" ) as file:
      if exec_lin_range:
         for datapoint in range( len(lin_spectrum) ):
            if lin_spectrum[datapoint][0] >= offset and lin_spectrum[datapoint][0] <= width:
               file.write( "{res:10.3f}{intense:15.6f}\n".format( res=(lin_spectrum[datapoint][0]), intense=lin_spectrum[datapoint][1] ) )
      else:
         for datapoint in range( len(lin_spectrum) ):
            file.write( "{res:10.3f}{intense:15.6f}\n".format( res=(lin_spectrum[datapoint][0]), intense=lin_spectrum[datapoint][1] ) )
#----------------------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------------------
if exec_baseline:

   spectrum = []

   try:
      with open(input_file, "r") as file:
         for line in file:
            line=line.strip()
            line=line.split()
            spectrum.append( [ float(line[0])*scale, float(line[1]) ] )
   except FileNotFoundError:
      print("\n Error: File '"+input_file+"' not found\n"); quit()


   intensities = []
   for vector in range( len(spectrum) ):
      intensities.append( spectrum[vector][1] )

   lamda = 1.0e+4
   pe = 1e-6
   if baseline_type == "als":
      baseline = BASELINE_ALS( intensities, lam=lamda, p=pe, niter=10 )
   if baseline_type == "arpls":
      baseline_tpm = BASELINE_ARPLS( intensities, lam=lamda, ratio=pe, niter=10, full_output=True )
      baseline = baseline_tpm[0].tolist()

   corrected_intensities = [ intensities[x] - baseline[x] for x in range( len(intensities) ) ]


   with open( input_file+".blc", "w" ) as file:
      for index in range( len(spectrum) ):
         file.write( "{freq:9.1f}{intense:16.8f}\n".format( freq=float(spectrum[index][0]), intense=corrected_intensities[index] ) )
#----------------------------------------------------------------------------------------------------------------------------



# Sort input file in reverse order
#----------------------------------------------------------------------------------------------------------------------------
if exec_invert:
   INVERT( input_file )
#----------------------------------------------------------------------------------------------------------------------------



