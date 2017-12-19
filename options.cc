// prototype for handling options as name=value pairs,
// either in a map or a vector.

#include <stdio.h>

#include <map>
#include <vector>

// -----------------------------------------------------------------------------
enum class Option {
    LookAhead,
    BlockSize,
    Tolerance,
};

class Value
{
public:
    Value( int i ) { i_ = i; }
    Value( double d ) { d_ = d; }

    int i_;
    double d_;
};

// -----------------------------------------------------------------------------
void use_map( std::map< Option, Value > opts )
{
    // set default values
    int lookahead = 2;
    int nb = 32;
    double tol = 1e-6;

    // override defaults
    for (auto opt = opts.begin(); opt != opts.end(); ++opt) {
        printf( "%s opt %d => %d\n", __func__, int(opt->first), opt->second.i_ );
        if (opt->first == Option::LookAhead) { lookahead = opt->second.i_; }
        if (opt->first == Option::BlockSize) { nb = opt->second.i_; }
        if (opt->first == Option::Tolerance) { tol = opt->second.d_; }
    }
    printf( "%s: lookahead %d, nb %d, tol %.2e\n",
            __func__, lookahead, nb, tol );
}

// -----------------------------------------------------------------------------
void use_vec( std::vector< std::pair< Option, Value > > opts )
{
    // set default values
    int lookahead = 2;
    int nb = 32;
    double tol = 1e-6;

    // override defaults
    for (auto opt = opts.begin(); opt != opts.end(); ++opt) {
        printf( "%s opt %d => %d\n", __func__, int(opt->first), opt->second.i_ );
        if (opt->first == Option::LookAhead) { lookahead = opt->second.i_; }
        if (opt->first == Option::BlockSize) { nb = opt->second.i_; }
        if (opt->first == Option::Tolerance) { tol = opt->second.d_; }
    }
    printf( "%s: lookahead %d, nb %d, tol %.2e\n",
            __func__, lookahead, nb, tol );
}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    use_map( { { Option::LookAhead, 2 },
               { Option::BlockSize, 32 } } );
    printf( "\n" );

    use_vec( { { Option::LookAhead, 2 },
               { Option::BlockSize, 32 },
               { Option::Tolerance, 1e-4 } } );
    return 0;
}
