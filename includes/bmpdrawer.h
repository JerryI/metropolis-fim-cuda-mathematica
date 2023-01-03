typedef struct {
    double r;       // a fraction between 0 and 1
    double g;       // a fraction between 0 and 1
    double b;       // a fraction between 0 and 1
} rgb;

typedef struct {
    double h;       // angle in degrees
    double s;       // a fraction between 0 and 1
    double v;       // a fraction between 0 and 1
} hsv;

static hsv   rgb2hsv(rgb in);
static rgb   hsv2rgb(hsv in);

hsv rgb2hsv(rgb in)
{
    hsv         out;
    double      min, max, delta;

    min = in.r < in.g ? in.r : in.g;
    min = min  < in.b ? min  : in.b;

    max = in.r > in.g ? in.r : in.g;
    max = max  > in.b ? max  : in.b;

    out.v = max;                                // v
    delta = max - min;
    if (delta < 0.00001)
    {
        out.s = 0;
        out.h = 0; // undefined, maybe nan?
        return out;
    }
    if( max > 0.0 ) { // NOTE: if Max is == 0, this divide would cause a crash
        out.s = (delta / max);                  // s
    } else {
        // if max is 0, then r = g = b = 0              
        // s = 0, h is undefined
        out.s = 0.0;
        out.h = NAN;                            // its now undefined
        return out;
    }
    if( in.r >= max )                           // > is bogus, just keeps compilor happy
        out.h = ( in.g - in.b ) / delta;        // between yellow & magenta
    else
    if( in.g >= max )
        out.h = 2.0 + ( in.b - in.r ) / delta;  // between cyan & yellow
    else
        out.h = 4.0 + ( in.r - in.g ) / delta;  // between magenta & cyan

    out.h *= 60.0;                              // degrees

    if( out.h < 0.0 )
        out.h += 360.0;

    return out;
}


rgb hsv2rgb(hsv in)
{
    double      hh, p, q, t, ff;
    long        i;
    rgb         out;

    if(in.s <= 0.0) {       // < is bogus, just shuts up warnings
        out.r = in.v;
        out.g = in.v;
        out.b = in.v;
        return out;
    }
    hh = in.h;
    if(hh >= 360.0) hh = 0.0;
    hh /= 60.0;
    i = (long)hh;
    ff = hh - i;
    p = in.v * (1.0 - in.s);
    q = in.v * (1.0 - (in.s * ff));
    t = in.v * (1.0 - (in.s * (1.0 - ff)));

    switch(i) {
    case 0:
        out.r = in.v;
        out.g = t;
        out.b = p;
        break;
    case 1:
        out.r = q;
        out.g = in.v;
        out.b = p;
        break;
    case 2:
        out.r = p;
        out.g = in.v;
        out.b = t;
        break;

    case 3:
        out.r = p;
        out.g = q;
        out.b = in.v;
        break;
    case 4:
        out.r = t;
        out.g = p;
        out.b = in.v;
        break;
    case 5:
    default:
        out.r = in.v;
        out.g = p;
        out.b = q;
        break;
    }
    return out;     
}

void write_bmp(const char *path, const uint width, const uint height, const uint8_t* const data) {
    const int pad=(4-(3*width)%4)%4, filesize=54+(3*width+pad)*height; // horizontal line must be a multiple of 4 bytes long, header is 54 bytes
    char header[54] = { 'B','M', 0,0,0,0, 0,0,0,0, 54,0,0,0, 40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,24,0 };
    for(uint i=0; i<4; i++) {
        header[ 2+i] = (char)((filesize>>(8*i))&255);
        header[18+i] = (char)((width   >>(8*i))&255);
        header[22+i] = (char)((height  >>(8*i))&255);
    }
    char* img = new char[filesize];
    for(uint i=0; i<54; i++) img[i] = header[i];
    for(uint y=0; y<height; y++) {
        for(uint x=0; x<width; x++) {

            const int i = 54+3*x+y*(3*width+pad);
            img[i  ] = (char)( data[(x+(height-1-y)*width)*3]);
            img[i+1] = (char)( data[(x+(height-1-y)*width)*3 + 1]);
            img[i+2] = (char)( data[(x+(height-1-y)*width)*3 + 2]);
        }
        for(uint p=0; p<pad; p++) img[54+(3*width+p)+y*(3*width+pad)] = 0;
    }

    std::ofstream file(path, std::ios::trunc);
    file.write(img, filesize);
    file.close();
    delete[] img;
}

void drawpicture(const char* path, int width, int height, int DF, float RADIUS, float* poso, float* spino) {
    unsigned long* indexes = new unsigned long[NS];
    unsigned long number = 0; 

    for(unsigned long i=0; i<NS; ++i) {
        if (( abs(poso[(i * 4) + 2] - 0.0f) < 2.0f ) && ( poso[(i * 4) + 3] > 0.0f ) ) {
            indexes[number] = i;
            ++number;
        }
    }

    std::cout << "number = " << number << "\n";

    float maxpx = -99999.0;
    float minpx =  99999.9;
    float maxpy = -99999.0;
    float minpy =  99999.9;    

    for (unsigned long i=0; i<number; ++i) {
        if (poso[(indexes[i] << 2)] > maxpx) maxpx = poso[(indexes[i] << 2)];
        if (poso[(indexes[i] << 2)] < minpx) minpx = poso[(indexes[i] << 2)];
        if (poso[(indexes[i] << 2) + 1] > maxpy) maxpy = poso[(indexes[i] << 2) + 1];
        if (poso[(indexes[i] << 2) + 1] < minpy) minpy = poso[(indexes[i] << 2) + 1];        
    }

    std::cout << "min x: " << minpx << "\n";
    std::cout << "min y: " << minpy << "\n";
    std::cout << "max x: " << maxpx << "\n";
    std::cout << "max y: " << maxpy << "\n";

    uint8_t* data = new uint8_t[width*height*3];

    memset(data, 0xFF, width*height*3);

    float angle;
    hsv pix; pix.s = 1.0; pix.v = 1.0;

    unsigned long index = 0;
    rgb npix;

    int ux, uy;
    int uux, uuy;
    
    int r,g,b;

    float ax, ay;

    for (unsigned long i=0; i<number; ++i) {
        index = indexes[i];
        pix.h = (180.0f/3.14159f)*atan2f(spino[(index << 2) + 1], spino[(index << 2)]);

        ax = poso[(index << 2)]     - minpx;
        ay = poso[(index << 2) + 1] - minpy;

        ax = ax / (maxpx - minpx);
        ay = ay / (maxpy - minpy);

        ux = trunc(((float)width)*ax);
        uy = trunc(((float)height)*ay);       


        npix = hsv2rgb(pix);
        npix.r *= 255.0f;
        npix.g *= 255.0f;
        npix.b *= 255.0f;



        //std::cout << "x = " << ux << "\n";
        //std::cout << "y = " << uy << "\n";
        rgb resulting_color;
        float alpha;

        for (int x=-DF; x<DF; ++x) {
          for (int y=-DF; y<DF; ++y) {

            uux = ux + x;
            uuy = uy + y;

            if (uux >= width)  uux = width-1;
            if (uuy >= height) uuy = height-1;

            if (uux < 0) uux = 0;
            if (uuy < 0) uuy = 0; 

            alpha = expf((float)(-(x*x)-(y*y))/RADIUS);

            if (data[(uux + (uuy * width))*3] == 0 && data[(uux + (uuy * width))*3 + 1] == 0 && data[(uux + (uuy * width))*3 + 2] == 0) {
                alpha = 1.0f;
            }

            resulting_color.r = (1.0f-alpha)*(float)data[(uux + (uuy * width))*3] + (alpha)*npix.r;
            resulting_color.g = (1.0f-alpha)*(float)data[(uux + (uuy * width))*3 + 1] + (alpha)*npix.g;
            resulting_color.b = (1.0f-alpha)*(float)data[(uux + (uuy * width))*3 + 2] + (alpha)*npix.b;

            r = trunc(resulting_color.r);
            b = trunc(resulting_color.b);
            g = trunc(resulting_color.g);

            if (r > 255) r = 255;
            if (g > 255) g = 255;
            if (b > 255) b = 255;

            if (r < 0) r = 0;
            if (g < 0) g = 0;
            if (b < 0) b = 0;

            data[(uux + (uuy * width))*3]     = (uint8_t) r;
            data[(uux + (uuy * width))*3 +1]  = (uint8_t) g;
            data[(uux + (uuy * width))*3 +2]  = (uint8_t) b;
          }
        }
    }

    write_bmp(path, width, height, (uint8_t*)data);

    free(data);
    free(indexes);
}