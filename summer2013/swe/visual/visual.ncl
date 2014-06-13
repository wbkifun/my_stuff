;**********************************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl" 
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl" 
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
;**********************************************************
 begin
;**********************************************************
; setup
;**********************************************************

 pi = 4*atan(1.)

 cs_ifile = "./uvh2.ascii"
 cs_data = asciiread(cs_ifile, (/5400,5/), "float")

 cs_lons = cs_data(:,0)*(180/pi)
 cs_lats = cs_data(:,1)*(180/pi)
 cs_u = cs_data(:,2)
 cs_v = cs_data(:,3)
 cs_h = cs_data(:,4)

;*************************************************
; vec = 0 means draw vector field
; vec = 1 means draw countour for variable h
;*************************************************
 vec = 0

if ( vec .eq. 0 ) then
;*************************************************
; Draw Map
;*************************************************

  wks = gsn_open_wks("ps","map")

  res = True
  res@gsnDraw      = False
  res@gsnFrame     = False
  ;res@gsnMaximize = True
  res@mpOutlineOn  = True
  res@mpDataBaseVersion               = "MediumRes"
  res@mpNationalLineThicknessF        = 0.1
  res@mpNationalLineColor             = 1
  res@mpLandFillColor                 = -1
  res@mpGeophysicalLineThicknessF     = .5
  res@mpNationalLineThicknessF        = .5

  plot = gsn_csm_map_ce(wks, res)

;*************************************************
;  Draw Vector
;*************************************************

  wmsetp("vch - head size", 0.002)
  wmsetp("vcc - vector color",2)
  wmsetp("vcw - vector linewidth scale factor", .05)
  wmsetp("vrs - vector length", 100.)
  wmvectmap(wks,cs_lats,cs_lons,cs_u,cs_v)

else

 wks = gsn_open_wks("ps","map")
 gsn_define_colormap(wks,"BlueWhiteOrangeRed")

 res                       = True
 res@gsnMaximize           = True
 res@gsnDraw               = False
 res@gsnFrame              = False
 res@cnFillOn              = True
 res@gsnSpreadColors       = True
 ;res@mpProjection          = "Orthographic"
 ;res@mpCenterLonF          = 128 
 ;res@mpCenterLatF          = 37
 res@cnLinesOn             = False
 res@lbLabelStride         = 5
 res@sfXArray              = cs_lons
 res@sfYArray              = cs_lats

;*************************************************
;  Draw contour
;*************************************************
 plot = gsn_csm_contour_map(wks,cs_h,res)

end if

 draw(plot)
 frame(wks)

end
