// Compile this file to an object file, e.g.:
//
//     gcc -O3 -c -o trace.o trace.c
//
// Then link it with your application.
//
// One easy way to use it with PLASMA is to put it in the link options.
// You can do it by adding the following line to your make.inc file:
//
//     LDFLAGS = -fopenmp tools/trace.o
//
// Then wrap the calls you want to trace with calls to trace_event_start()
// and trace_event_stop(), e.g.:
//
//     trace_cpu_start();
//     cblas_zgemm(CblasColMajor, ...
//     trace_cpu_stop("LightGoldenrodYellow");
//
// Provide the name of the color as a string.
// Use one of the X11 color names: https://en.wikipedia.org/wiki/X11_color_names
//
// Optionally, you can assign a label to a color using trace_label(), e.g.:
//
//      trace_label("Teal", "gemm");
//
// Upon completion, the trace is written to an SVG file in the local folder.
// The name has the form trace_189648000.svg, where the number is the Unix time.
//
// Initially, tracing is on.
// You can turn it off by calling tracing_off();
// You can turn it back on by calling tracing_on();
//
// Unlike in the past renditions of this solution, here:
// - you do not include a header file,
// - you do not provide the color as an integer, but as a string,
// - you do not call the constructor trace_init(),
// - you do not call the destructor trace_finish().

#include <omp.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>
#include <string.h>

// https://en.wikipedia.org/wiki/X11_color_names
static struct {
    const char *color;
    int value;
} Color[] = {
    /* Pink colors */                   /* Cyan colors */
    {"Pink",                 0xFFC0CB},   {"Aqua",            0x00FFFF},
    {"LightPink",            0xFFB6C1},   {"Cyan",            0x00FFFF},
    {"HotPink",              0xFF69B4},   {"LightCyan",       0xE0FFFF},
    {"DeepPink",             0xFF1493},   {"PaleTurquoise",   0xAFEEEE},
    {"PaleVioletRed",        0xDB7093},   {"Aquamarine",      0x7FFFD4},
    {"MediumVioletRed",      0xC71585},   {"Turquoise",       0x40E0D0},
                                          {"MediumTurquoise", 0x48D1CC},
    /* Red colors */                      {"DarkTurquoise",   0x00CED1},
    {"LightSalmon",          0xFFA07A},   {"LightSeaGreen",   0x20B2AA},
    {"Salmon",               0xFA8072},   {"CadetBlue",       0x5F9EA0},
    {"DarkSalmon",           0xE9967A},   {"DarkCyan",        0x008B8B},
    {"LightCoral",           0xF08080},   {"Teal",            0x008080},
    {"IndianRed",            0xCD5C5C},
    {"Crimson",              0xDC143C},   /* Blue colors */
    {"FireBrick",            0xB22222},   {"LightSteelBlue",  0xB0C4DE},
    {"DarkRed",              0x8B0000},   {"PowderBlue",      0xB0E0E6},
    {"Red",                  0xFF0000},   {"LightBlue",       0xADD8E6},
                                          {"SkyBlue",         0x87CEEB},
    /* Orange colors */                   {"LightSkyBlue",    0x87CEFA},
    {"OrangeRed",            0xFF4500},   {"DeepSkyBlue",     0x00BFFF},
    {"Tomato",               0xFF6347},   {"DodgerBlue",      0x1E90FF},
    {"Coral",                0xFF7F50},   {"CornflowerBlue",  0x6495ED},
    {"DarkOrange",           0xFF8C00},   {"SteelBlue",       0x4682B4},
    {"Orange",               0xFFA500},   {"RoyalBlue",       0x4169E1},
                                          {"Blue",            0x0000FF},
    /* Yellow colors */                   {"MediumBlue",      0x0000CD},
    {"Yellow",               0xFFFF00},   {"DarkBlue",        0x00008B},
    {"LightYellow",          0xFFFFE0},   {"Navy",            0x000080},
    {"LemonChiffon",         0xFFFACD},   {"MidnightBlue",    0x191970},
    {"LightGoldenrodYellow", 0xFAFAD2},
    {"PapayaWhip",           0xFFEFD5},   /* Purple, violet, and magenta */
    {"Moccasin",             0xFFE4B5},   {"Lavender",        0xE6E6FA},
    {"PeachPuff",            0xFFDAB9},   {"Thistle",         0xD8BFD8},
    {"PaleGoldenrod",        0xEEE8AA},   {"Plum",            0xDDA0DD},
    {"Khaki",                0xF0E68C},   {"Violet",          0xEE82EE},
    {"DarkKhaki",            0xBDB76B},   {"Orchid",          0xDA70D6},
    {"Gold",                 0xFFD700},   {"Fuchsia",         0xFF00FF},
                                          {"Magenta",         0xFF00FF},
    /* Brown colors */                    {"MediumOrchid",    0xBA55D3},
    {"Cornsilk",             0xFFF8DC},   {"MediumPurple",    0x9370DB},
    {"BlanchedAlmond",       0xFFEBCD},   {"BlueViolet",      0x8A2BE2},
    {"Bisque",               0xFFE4C4},   {"DarkViolet",      0x9400D3},
    {"NavajoWhite",          0xFFDEAD},   {"DarkOrchid",      0x9932CC},
    {"Wheat",                0xF5DEB3},   {"DarkMagenta",     0x8B008B},
    {"BurlyWood",            0xDEB887},   {"Purple",          0x800080},
    {"Tan",                  0xD2B48C},   {"Indigo",          0x4B0082},
    {"RosyBrown",            0xBC8F8F},   {"DarkSlateBlue",   0x483D8B},
    {"SandyBrown",           0xF4A460},   {"SlateBlue",       0x6A5ACD},
    {"Goldenrod",            0xDAA520},   {"MediumSlateBlue", 0x7B68EE},
    {"DarkGoldenrod",        0xB8860B},
    {"Peru",                 0xCD853F},   /* White colors */
    {"Chocolate",            0xD2691E},   {"White",           0xFFFFFF},
    {"SaddleBrown",          0x8B4513},   {"Snow",            0xFFFAFA},
    {"Sienna",               0xA0522D},   {"Honeydew",        0xF0FFF0},
    {"Brown",                0xA52A2A},   {"MintCream",       0xF5FFFA},
    {"Maroon",               0x800000},   {"Azure",           0xF0FFFF},
                                          {"AliceBlue",       0xF0F8FF},
    /* Green colors */                    {"GhostWhite",      0xF8F8FF},
    {"DarkOliveGreen",       0x556B2F},   {"WhiteSmoke",      0xF5F5F5},
    {"Olive",                0x808000},   {"Seashell",        0xFFF5EE},
    {"OliveDrab",            0x6B8E23},   {"Beige",           0xF5F5DC},
    {"YellowGreen",          0x9ACD32},   {"OldLace",         0xFDF5E6},
    {"LimeGreen",            0x32CD32},   {"FloralWhite",     0xFFFAF0},
    {"Lime",                 0x00FF00},   {"Ivory",           0xFFFFF0},
    {"LawnGreen",            0x7CFC00},   {"AntiqueWhite",    0xFAEBD7},
    {"Chartreuse",           0x7FFF00},   {"Linen",           0xFAF0E6},
    {"GreenYellow",          0xADFF2F},   {"LavenderBlush",   0xFFF0F5},
    {"SpringGreen",          0x00FF7F},   {"MistyRose",       0xFFE4E1},
    {"MediumSpringGreen",    0x00FA9A},
    {"LightGreen",           0x90EE90},   /* Gray and black colors */
    {"PaleGreen",            0x98FB98},   {"Gainsboro",       0xDCDCDC},
    {"DarkSeaGreen",         0x8FBC8F},   {"LightGray",       0xD3D3D3},
    {"MediumAquamarine",     0x66CDAA},   {"Silver",          0xC0C0C0},
    {"MediumSeaGreen",       0x3CB371},   {"DarkGray",        0xA9A9A9},
    {"SeaGreen",             0x2E8B57},   {"Gray",            0x808080},
    {"ForestGreen",          0x228B22},   {"DimGray",         0x696969},
    {"Green",                0x008000},   {"LightSlateGray",  0x778899},
    {"DarkGreen",            0x006400},   {"SlateGray",       0x708090},
                                          {"DarkSlateGray",   0x2F4F4F},
                                          {"Black",           0x000000}
};

static int Trace = 1;
void trace_off() {Trace = 0;}
void trace_on()  {Trace = 1;}

#define IMAGE_WIDTH 2390
#define IMAGE_HEIGHT 1000

#define MAP_SIZE 1024
static int ColorMap[MAP_SIZE];
static const char *Label[sizeof(Color)/sizeof(Color[0])] = { NULL };
static int NumColors = sizeof(Color)/sizeof(Color[0]);

static int NumThreads;
#define MAX_THREADS 256
#define MAX_THREAD_EVENTS 65536
static int    EventNumThread  [MAX_THREADS];
static double EventStartThread[MAX_THREADS][MAX_THREAD_EVENTS];
static double EventStopThread [MAX_THREADS][MAX_THREAD_EVENTS];
static int    EventColorThread[MAX_THREADS][MAX_THREAD_EVENTS];

//------------------------------------------------------------------------------
// https://en.wikipedia.org/wiki/Fowler–Noll–Vo_hash_function
static inline unsigned int color_index(const char *str)
{
    unsigned int hash = 23;
    unsigned int c;
    unsigned char *ustr = (unsigned char*)str;
    while ((c = *ustr++) != '\0')
        hash = hash*307+c;
    return hash%MAP_SIZE;
}

//------------------------------------------------------------------------------
void trace_cpu_start()
{
    int thread_num = omp_get_thread_num() & (MAX_THREADS-1);
    thread_num &= (MAX_THREADS-1);

    int event_num = EventNumThread[thread_num];
    EventStartThread[thread_num][event_num] = omp_get_wtime();
}

//------------------------------------------------------------------------------
void trace_cpu_stop(const char *color)
{
    int thread_num = omp_get_thread_num();
    thread_num &= (MAX_THREADS-1);

    int event_num = EventNumThread[thread_num];
    EventStopThread[thread_num][event_num] = omp_get_wtime();
    EventColorThread[thread_num][event_num] = ColorMap[color_index(color)];

    EventNumThread[thread_num] += Trace;
    EventNumThread[thread_num] &= (MAX_THREAD_EVENTS-1);
}

//------------------------------------------------------------------------------
void trace_label(const char *color, const char *label)
{
    Label[ColorMap[color_index(color)]] = label;
}

//------------------------------------------------------------------------------
static void trace_finish()
{
    double min_time = INFINITY;
    double max_time = 0.0;

    for (int thread = 0; thread < NumThreads; thread++)
        if (EventNumThread[thread] > 0)
            if (EventStartThread[thread][0] < min_time)
                min_time = EventStartThread[thread][0];

    for (int thread = 0; thread < NumThreads; thread++)
        if (EventNumThread[thread] > 0)
            if (EventStopThread[thread][EventNumThread[thread]-1] > max_time)
                max_time = EventStopThread[thread][EventNumThread[thread]-1];

    double total_time = max_time - min_time;
    double hscale = IMAGE_WIDTH / total_time;
    double vscale = IMAGE_HEIGHT / (NumThreads + 1);

    char file_name[32];
    snprintf(file_name, 32, "trace_%ld.svg", (unsigned long int)time(NULL));
    FILE *trace_file = fopen(file_name, "w");
    assert(trace_file != NULL);

    fprintf(trace_file,
            "<svg viewBox=\"0 0 %d %d\">\n", IMAGE_WIDTH, IMAGE_HEIGHT);

    // output events
    int thread;
    int event;
    for (thread = 0; thread < NumThreads; thread++) {
        for (event = 0; event < EventNumThread[thread]; event++) {
            double start = EventStartThread[thread][event]-min_time;
            double stop = EventStopThread[thread][event]-min_time;
            fprintf(
                trace_file,
                "<rect x=\"%lf\" y=\"%lf\" width=\"%lf\" height=\"%lf\" "
                "fill=\"#%06x\" stroke=\"#000000\" stroke-width=\"0.2\" "
                "inkscape:label=\"%s\"/>\n",
                start * hscale,
                thread * vscale,
                (stop-start) * hscale,
                0.9 * vscale,
                Color[EventColorThread[thread][event]].value,
                Label[EventColorThread[thread][event]]);
        }
    }

    // output legend
    int x = 0;
    int y = IMAGE_HEIGHT+50;
    for (int color = 0; color < NumColors; color++) {
        if (Label[color] != NULL) {
            fprintf(
                trace_file,
                "<rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" "
                "fill=\"#%06x\" stroke=\"#000000\" stroke-width=\"1\"/>\n"
                "<text x=\"%d\" y=\"%d\" "
                "font-family=\"monospace\" font-size=\"35\" fill=\"black\">"
                "%s</text>\n",
                x, y,
                50, 50,
                Color[color].value,
                x+75, y+36,
                Label[color]);
            x += 150;
            x += strlen(Label[color])*22;
            if (x > IMAGE_WIDTH) {
                x = 0;
                y += 100;
            }
        }
    }

    // output xticks time scale
    // xtick spacing is power of 10, with at most 20 tick marks
    double pwr = ceil( log10( total_time / 20 ));
    double xtick = pow( 10., pwr );
    int decimal_places = (pwr < 0 ? (int)-pwr : 0);
    for (double t = 0; t < total_time; t += xtick) {
        fprintf(
            trace_file,
            "<line x1=\"%f\" x2=\"%f\" y1=\"%f\" y2=\"%f\" "
            "stroke=\"#000000\" stroke-width=\"1\" />\n"
            "<text x=\"%f\" y=\"%f\" "
            "font-family=\"monospace\" font-size=\"35\">%.*f</text>\n",
            hscale * t,
            hscale * t,
            vscale * NumThreads,
            vscale * (NumThreads + 0.9),
            hscale * (t + 0.05*xtick),
            vscale * (NumThreads + 0.9),
            decimal_places, t);
    }

    fprintf(trace_file, "</svg>\n");
    fclose(trace_file);
    fprintf(stderr, "trace file: %s\n", file_name);
}

//------------------------------------------------------------------------------
__attribute__ ((constructor))
static void trace_init()
{
    // Check if the maximums are powers of two.
    assert (__builtin_popcount(MAX_THREADS) == 1);
    assert (__builtin_popcount(MAX_THREAD_EVENTS) == 1);

    // Initialize the color map.
    for (int i = 0; i < NumColors; i++)
        ColorMap[color_index(Color[i].color)] = i;

    // Clip the number of threads.
    NumThreads = omp_get_max_threads() < MAX_THREADS ?
        omp_get_max_threads() : MAX_THREADS;

    // Register the destructor.
    atexit(trace_finish);
}
