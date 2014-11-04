/*
station2modelgrid.c
Luiz Rodrigo Tozzi - luizrodrigotozzi@gmail.com
*/

#include <stdio.h>

/* Structure that describes a report header in a stn file */
struct rpthdr {
  char  id[8];    /* Station ID */
  float lat;      /* Latitude of Station */
  float lon;      /* Longitude of Station */
  float t;        /* Time in grid-relative units */
  int   nlev;	  /* Number of levels following */
  int   flag;     /* Level independent var set flag */
} hdr;

main ()
{
  FILE  *ifile, *ofile;
  char  rec[80];
  int   flag,year,month,day,hour,yrsav,mnsav,ddsav,hhsav,i;
  float val;

/* Open files */
  ifile = fopen ("station2modelgrid.txt","r");
  ofile = fopen ("station2modelgrid.bin","wb");
  if (ifile==NULL || ofile==NULL) {
    printf("Error opening files\n");
    return;
  }

/* Read, write loop */
  flag = 1;
  while (fgets(rec,79,ifile)!=NULL) {
    /* Format conversion */
    sscanf (rec,"%i %i %i %i",&year,&month,&day,&hour);
    sscanf (rec+26," %g %g %g",&hdr.lat,&hdr.lon,&val);
    for (i=0; i<8; i++) hdr.id[i] = rec[i+11];
    /* Time group terminator if needed */
    if (flag) {
      yrsav = year;
      mnsav = month;
      ddsav = day;
      hhsav = hour;
      flag = 0;
    }
    if (yrsav!=year || mnsav!=month || ddsav!=day) {
      hdr.nlev = 0;
      fwrite(&hdr,sizeof(struct rpthdr), 1, ofile);
    }
    yrsav = year;
    mnsav = month;
    ddsav = day;
    hhsav = hour;
    /* Write this report */
    hdr.nlev = 1;
    hdr.flag = 1;
    hdr.t = 0.0;
    fwrite (&hdr,sizeof(struct rpthdr), 1, ofile);
    fwrite (&val,sizeof(float), 1, ofile);
  }
  hdr.nlev = 0;
  fwrite (&hdr,sizeof(struct rpthdr), 1, ofile);
  }
