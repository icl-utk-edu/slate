//------------------------------------------------------------------------------
// Copyright (c) 2017, University of Tennessee
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the University of Tennessee nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//------------------------------------------------------------------------------
// This research was supported by the Exascale Computing Project (17-SC-20-SC),
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
//------------------------------------------------------------------------------
// Need assistance with the SLATE software? Join the "SLATE User" Google group
// by going to https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user
// and clicking "Apply to join group". Upon acceptance, email your questions and
// comments to <slate-user@icl.utk.edu>.
//------------------------------------------------------------------------------

///-----------------------------------------------------------------------------
/// \file
///
/// prototype for handling options as name=value pairs,
/// either in a map or a vector.
///
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

//------------------------------------------------------------------------------
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

//------------------------------------------------------------------------------
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
