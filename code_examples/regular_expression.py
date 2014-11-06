import re


#------------------------------------------------------------------------------
# Extract number between string
#------------------------------------------------------------------------------
# () indicate the target string which will be extracted
# [] set of a character or a digit
# + repeat >= 1  cf) * >= 0, ? 0 or 1, {m} m-th repeat
#------------------------------------------------------------------------------
s1 = 'cs_grid_coords_ne30ngq4.nc'
ne = int( re.search('ne([0-9]+)',s1).group(1) )
ngq = int( re.search('ngq([0-9]+)',s1).group(1) )
print ne, ngq


# other method
ne2, ngq2 = [int(c) for c in re.findall('ne([0-9]+)ngq([0-9]+)',s1)[0]]
print ne2, ngq2
