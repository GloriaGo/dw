#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "dtypes.h"
#include "utils.h"
#include "ieee754.h"

int double_exponent(double d)
{
    union ieee754_double dl;

    dl.d = d;
    return dl.ieee.exponent - IEEE754_DOUBLE_BIAS;
}

void snprint_real(char *s, size_t cnt, double r,
                  int width,    // printf field width, or 0
                  int dec,      // # decimal digits desired, recommend 16
                  // # of zeros in .00...0x before using scientific notation
                  // recommend 3-4 or so
                  int max_digs_rt,
                  // # of digits left of decimal before scientific notation
                  // recommend 10
                  int max_digs_lf)
{
    int mag;
    double fpart, temp;
    char format[8];
    char num_format[3];
    int sz, keepz=0;

    s[0] = '\0';
    if (width == -1) {
        width = 0;
        keepz=1;
    }
    if (isnan(r)) {
        if (sign_bit(r))
            strncpy(s, "-nan", cnt);
        else
            strncpy(s, "nan", cnt);
        return;
    }
    if (r == 0) {
        strncpy(s, "0", cnt);
        return;
    }

    num_format[0] = 'l';
    num_format[2] = '\0';

    mag = double_exponent(r);

    mag = (int)(((double)mag)/LOG2_10 + 0.5);
    if (r == 0)
        mag = 0;
    if ((mag > max_digs_lf-1) || (mag < -max_digs_rt)) {
        num_format[1] = 'e';
        temp = r/pow(10, mag);      /* see if number will have a decimal */
        fpart = temp - floor(temp); /* when written in scientific notation */
    }
    else {
        num_format[1] = 'f';
        fpart = r - floor(r);
    }
    if (fpart == 0)
        dec = 0;
    if (width == 0) {
        snprintf(format, 8, "%%.%d%s", dec, num_format);
    }
    else {
        snprintf(format, 8, "%%%d.%d%s", width, dec, num_format);
    }
    sz = snprintf(s, cnt, format, r);
    /* trim trailing zeros from fractions. not when using scientific
       notation, since we might have e.g. 1.2000e+100. also not when we
       need a specific output width */
    if (width == 0 && !keepz) {
        if (sz > 2 && fpart && num_format[1]!='e') {
            while (s[sz-1] == '0') {
                s[sz-1]='\0';
                sz--;
            }
            // don't need trailing .
            if (s[sz-1] == '.') {
                s[sz-1] = '\0';
                sz--;
            }
        }
    }
    // TODO. currently 1.1e20 prints as 1.1000000000000000e+20; be able to
    // get rid of all those zeros.
}

void snprint_cplx(char *s, size_t cnt, double re, double im,
                  // args to pass on to snprint_real
                  int width, int dec,
                  int max_digs_rt, int max_digs_lf,
                  // print spaces around sign in a+bi
                  int spflag)
{
    int fzr = (re==0) || rel_zero(re,im);
    int fzi = (im==0) || rel_zero(im,re);
    size_t len, sl;
    size_t space = cnt;

    s[0] = '\0';
    if (isnan(im) && fzr) {
        if (space < 2) return;
        snprint_real(s, space-2, im, width, dec, max_digs_rt, max_digs_lf);
        strcat(s, "i");
        return;
    }
    if (!fzr || (fzr && fzi)) {
        if (space < 4) return;
        snprint_real(s, space-4, re, width, dec, max_digs_rt, max_digs_lf);
        if ((im >= 0 || (isnan(im)&&!sign_bit(im))) && !fzi) {
            if (spflag) {
                strcat(s, " + ");
            }
            else {
                strcat(s, "+");
            }
        }
        else if (!fzi) {
            im = -im;
            if (spflag)
                strcat(s, " - ");
            else
                strcat(s, "-");
        }
    }
    if (!fzi) {
        len = sl = strlen(s);
        if (im == -1) {
            while ((long)(len-sl) < (long)(width-2) && len < (space-3))
                s[len++] = ' ';
            s[len] =   '-';
            s[len+1] = 'i';
            s[len+2] = '\0';
        }
        else if (im == 1) {
            while ((long)(len-sl) < (long)(width-1) && len < (space-2))
                s[len++] = ' ';
            s[len] =   'i';
            s[len+1] = '\0';
        }
        else {
            snprint_real(s+len, space-len-2, im, width, dec,
                         max_digs_rt, max_digs_lf);
            strcat(s, "i");
        }
    }
}
