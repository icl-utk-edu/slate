#!/usr/bin/env python3
#
# Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
#
# Derived from plasma/tools/fortran_gen.py

from __future__ import print_function

import sys
import os
import re
import argparse

description = '''\
Generates Fortran 2003 interface from SLATE C API header files.'''

help = '''\
----------------------------------------------------------------------
Example uses:

  generate_fortran.py slate/c_api/*.h
      generates slate_module.f90 with module slate

----------------------------------------------------------------------
'''

#-------------------------------------------------------------
# command line options
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=description,
    epilog=help )
parser.add_argument('--output',        action='store', help='Output file', default='slate_module.f90')
parser.add_argument('args', nargs='*', action='store', help='Files to process')
opts = parser.parse_args()

#-------------------------------------------------------------
# set indentation in the f90 file
tab = "    "
indent = tab

module_name = "slate"
mpi_header = "use mpi"
# mpi_header = "use mpi_f08" todo

# translation_table of types
types_dict = {
    "int":                             ("integer(kind=c_int)"),
    "int64_t":                         ("integer(kind=c_int64_t)"),
    "double":                          ("real(kind=c_double)"),
    "float":                           ("real(kind=c_float)"),
    "double _Complex":                 ("complex(kind=c_double_complex)"),
    "float _Complex":                  ("complex(kind=c_float_complex)"),
    "slate_Norm":                      ("character(kind=c_char)"),
    "slate_Matrix_r32":                ("type(c_ptr)"),
    "slate_Matrix_r64":                ("type(c_ptr)"),
    "slate_Matrix_c32":                ("type(c_ptr)"),
    "slate_Matrix_c64":                ("type(c_ptr)"),
    "slate_BandMatrix_r32":            ("type(c_ptr)"),
    "slate_BandMatrix_r64":            ("type(c_ptr)"),
    "slate_BandMatrix_c32":            ("type(c_ptr)"),
    "slate_BandMatrix_c64":            ("type(c_ptr)"),
    "slate_HermitianMatrix_r32":       ("type(c_ptr)"),
    "slate_HermitianMatrix_r64":       ("type(c_ptr)"),
    "slate_HermitianMatrix_c32":       ("type(c_ptr)"),
    "slate_HermitianMatrix_c64":       ("type(c_ptr)"),
    "slate_HermitianBandMatrix_r32":   ("type(c_ptr)"),
    "slate_HermitianBandMatrix_r64":   ("type(c_ptr)"),
    "slate_HermitianBandMatrix_c32":   ("type(c_ptr)"),
    "slate_HermitianBandMatrix_c64":   ("type(c_ptr)"),
    "slate_TriangularMatrix_r32":      ("type(c_ptr)"),
    "slate_TriangularMatrix_r64":      ("type(c_ptr)"),
    "slate_TriangularMatrix_c32":      ("type(c_ptr)"),
    "slate_TriangularMatrix_c64":      ("type(c_ptr)"),
    "slate_TriangularBandMatrix_r32":  ("type(c_ptr)"),
    "slate_TriangularBandMatrix_r64":  ("type(c_ptr)"),
    "slate_TriangularBandMatrix_c32":  ("type(c_ptr)"),
    "slate_TriangularBandMatrix_c64":  ("type(c_ptr)"),
    "slate_SymmetricMatrix_r32":       ("type(c_ptr)"),
    "slate_SymmetricMatrix_r64":       ("type(c_ptr)"),
    "slate_SymmetricMatrix_c32":       ("type(c_ptr)"),
    "slate_SymmetricMatrix_c64":       ("type(c_ptr)"),
    "slate_TrapezoidMatrix_r32":       ("type(c_ptr)"),
    "slate_TrapezoidMatrix_r64":       ("type(c_ptr)"),
    "slate_TrapezoidMatrix_c32":       ("type(c_ptr)"),
    "slate_TrapezoidMatrix_c64":       ("type(c_ptr)"),
    "slate_Pivots":                    ("type(c_ptr)"),
    "slate_TriangularFactors_r32":     ("type(c_ptr)"),
    "slate_TriangularFactors_r64":     ("type(c_ptr)"),
    "slate_TriangularFactors_c32":     ("type(c_ptr)"),
    "slate_TriangularFactors_c64":     ("type(c_ptr)"),
    "slate_OptionValue":               ("type(slate_OptionValue)"),
    # "slate_Options":                   ("type(slate_Options)"),
    "slate_Options":                   ("type(c_ptr)"),
    "slate_Option":                    ("type(slate_Option)"),
    "slate_Side":                      ("character(kind=c_char)"),
    "slate_Diag":                      ("character(kind=c_char)"),
    "slate_Op":                        ("character(kind=c_char)"),
    "slate_Uplo":                      ("character(kind=c_char)"),
    "slate_Layout":                    ("character(kind=c_char)"),
    "slate_TileKind":                  ("integer(kind=c_int)"),
    "slate_Tile_r32":                  ("type(slate_Tile_r32)"),
    "slate_Tile_r64":                  ("type(slate_Tile_r64)"),
    "slate_Tile_c32":                  ("type(slate_Tile_c32)"),
    "slate_Tile_c64":                  ("type(slate_Tile_c64)"),
    "MPI_Comm":                        ("integer(kind=c_int)"),
    "MPI_Fint":                        ("integer(kind=c_int)"),
    # "MPI_Comm":                        ("type(MPI_Comm)"),
    "bool":                            ("logical(kind=c_bool)"),
    "void":                            ("type(c_ptr)"),
}

# translation_table with names of auxiliary variables
return_variables_dict = {}

# name arrays which will be translated to assumed-size arrays, e.g. pA(*)
arrays_names_2D = []
arrays_names_1D = ["A", "Sigma", "Lambda"]

# exclude inline functions from the interface
exclude_list = [
    "inline",

    "slate_Matrix_create_r32",
    "slate_Matrix_create_r64",
    "slate_Matrix_create_c32",
    "slate_Matrix_create_c64",
    "slate_Matrix_create_fromScaLAPACK_r32",
    "slate_Matrix_create_fromScaLAPACK_r64",
    "slate_Matrix_create_fromScaLAPACK_c32",
    "slate_Matrix_create_fromScaLAPACK_c64",

    "slate_BandMatrix_create_r32",
    "slate_BandMatrix_create_r64",
    "slate_BandMatrix_create_c32",
    "slate_BandMatrix_create_c64",

    "slate_HermitianMatrix_create_r32",
    "slate_HermitianMatrix_create_r64",
    "slate_HermitianMatrix_create_c32",
    "slate_HermitianMatrix_create_c64",
    "slate_HermitianMatrix_create_fromScaLAPACK_r32",
    "slate_HermitianMatrix_create_fromScaLAPACK_r64",
    "slate_HermitianMatrix_create_fromScaLAPACK_c32",
    "slate_HermitianMatrix_create_fromScaLAPACK_c64",

    "slate_HermitianBandMatrix_create_r32",
    "slate_HermitianBandMatrix_create_r64",
    "slate_HermitianBandMatrix_create_c32",
    "slate_HermitianBandMatrix_create_c64",

    "slate_SymmetricMatrix_create_r32",
    "slate_SymmetricMatrix_create_r64",
    "slate_SymmetricMatrix_create_c32",
    "slate_SymmetricMatrix_create_c64",
    "slate_SymmetricMatrix_create_fromScaLAPACK_r32",
    "slate_SymmetricMatrix_create_fromScaLAPACK_r64",
    "slate_SymmetricMatrix_create_fromScaLAPACK_c32",
    "slate_SymmetricMatrix_create_fromScaLAPACK_c64",

    "slate_TriangularMatrix_create_r32",
    "slate_TriangularMatrix_create_r64",
    "slate_TriangularMatrix_create_c32",
    "slate_TriangularMatrix_create_c64",
    "slate_TriangularMatrix_create_fromScaLAPACK_r32",
    "slate_TriangularMatrix_create_fromScaLAPACK_r64",
    "slate_TriangularMatrix_create_fromScaLAPACK_c32",
    "slate_TriangularMatrix_create_fromScaLAPACK_c64",

    "slate_TriangularBandMatrix_create_r32",
    "slate_TriangularBandMatrix_create_r64",
    "slate_TriangularBandMatrix_create_c32",
    "slate_TriangularBandMatrix_create_c64",

    "slate_TrapezoidMatrix_create_r32",
    "slate_TrapezoidMatrix_create_r64",
    "slate_TrapezoidMatrix_create_c32",
    "slate_TrapezoidMatrix_create_c64",
    "slate_TrapezoidMatrix_create_fromScaLAPACK_r32",
    "slate_TrapezoidMatrix_create_fromScaLAPACK_r64",
    "slate_TrapezoidMatrix_create_fromScaLAPACK_c32",
    "slate_TrapezoidMatrix_create_fromScaLAPACK_c64",

]

#-------------------------------------------------------------

# global list used to determine derived types
derived_types = []

#-------------------------------------------------------------------------------
def parse_triple(string):
    """Parse string of
       type (*)name
       into triple of [type, pointer, name]"""

    string = string.strip()
    if "_Complex" in string:
        if (string.find("**") > -1):
            parts = string.split(" _Complex** ")
            parts[0] += " _Complex**"
        elif (string.find("*") > -1):
            parts = string.split(" _Complex* ")
            parts[0] += " _Complex*"
        else:
            parts = string.split(" _Complex ")
            parts[0] += " _Complex"
    else:
        parts = string.split()

    if (len(parts) < 2 or len(parts) > 3):
        print("Error: Cannot detect type for ", string)

    type_part = str.strip(parts[0])

    if (len(parts) == 2):
        name_with_pointer = str.strip(parts[1])
        if (name_with_pointer.find("**") > -1):
            pointer_part = "**"
            name_part = name_with_pointer.replace("**", "")
        elif (name_with_pointer.find("*") > -1):
            pointer_part = "*"
            name_part    = name_with_pointer.replace("*", "")
        else:
            pointer_part = ""
            name_part    = name_with_pointer

    elif (len(parts) == 3):
        if (str.strip(parts[1]) == "**"):
            pointer_part = "**"
            name_part    = str.strip(parts[2])
        elif (str.strip(parts[1]) == "*"):
            pointer_part = "*"
            name_part    = str.strip(parts[2])
        else:
            print("Error: Too many parts for ", string)

    name_part = name_part.strip()

    return [type_part, pointer_part, name_part]
# end

#-------------------------------------------------------------------------------
def iso_c_interface_type(arg, return_value):
    """Generate a declaration for a variable in the interface."""

    if "*" in arg[0]:
        arg[0] = arg[0].split("*")[0]
        arg[1] = "*"

    if (arg[1] == "*" or arg[1] == "**"):
        is_pointer = True
    else:
        is_pointer = False

    if (is_pointer):
        f_type = "type(c_ptr)"
    else:
        f_type = types_dict[arg[0]]

    if (not return_value and arg[1] != "**" and arg[1] != "*"):
        f_pointer = ", value"
    else:
        f_pointer = ""

    f_name = arg[2]

    if (is_pointer and f_name in arrays_names_1D):
        f_pointer = ", value"

    f_line = f_type + f_pointer + " :: " + f_name

    return f_line
# end

#-------------------------------------------------------------------------------
def iso_c_wrapper_type(arg):
    """Generate a declaration for a variable in the Fortran wrapper."""

    if (arg[1] == "*" or arg[1] == "**"):
        is_pointer = True
    else:
        is_pointer = False

    if (is_pointer and arg[0].strip() == "void"):
        f_type = "type(c_ptr)"
    else:
        f_type = types_dict[arg[0]]

    #if (is_pointer):
    #    f_intent = ", intent(inout)"
    #else:
    #    f_intent = ", intent(in)"

    if (is_pointer):
        if (arg[1] == "*"):
           f_target = ", target"
        else:
           f_target = ", pointer"
    else:
        f_target = ""

    f_name    = arg[2]

    # detect array argument
    if   (is_pointer and f_name in arrays_names_2D):
        f_array = "(*)"
    elif (is_pointer and f_name in arrays_names_1D):
        f_array = "(*)"
    else:
        f_array = ""

    if (f_type == 'character(kind=c_char)'):
        f_target = ", value"

    #f_line = f_type + f_intent + f_target + " :: " + f_name + f_array
    f_line = f_type + f_target + " :: " + f_name + f_array

    return f_line
# end

#-------------------------------------------------------------------------------
def fortran_interface_enum(enum):
    """Generate an interface for an enum.
       Translate it into constants."""

    # initialize a string with the fortran interface
    f_interface = ""

    # loop over the arguments of the enum
    for param in enum:
        name  = param[0]
        value = param[1]

        type = ""
        if re.search(r'\d', value):
            if name == "slate_Norm_One" or name == "slate_Norm_Two":
                type = "character"
            else:
                type = "integer"
        else:
            type = "character"
        f_interface += indent + type + ", parameter :: " + name + " = " + value + "\n"

    return f_interface
# end

#-------------------------------------------------------------------------------
def fortran_interface_struct(struct):
    """Generate an interface for a struct.
       Translate it into a derived type."""

    # initialize a string with the fortran interface
    f_interface = ""

    name = struct[0][2]
    f_interface += tab + "type, bind(c) :: " + name + "\n"
    # loop over the arguments of the enum
    for j in range(1,len(struct)):
        f_interface += indent + tab + iso_c_interface_type(struct[j], True)
        f_interface += "\n"

    f_interface += tab + "end type " + name + "\n"

    return f_interface
# end

#-------------------------------------------------------------------------------
def fortran_interface_function(function):
    """Generate an interface for a function."""

    # is it a function or a subroutine
    if (function[0][0] == "void"):
        is_function = False
    else:
        is_function = True

    c_symbol = function[0][2]
    f_symbol = c_symbol + "_c"

    used_derived_types = set([])
    for arg in function:
        type_name = arg[0]
        if (type_name in derived_types):
            used_derived_types.add(type_name)

    # initialize a string with the fortran interface
    f_interface = ""
    f_interface += indent + "interface\n"

    if (is_function):
        f_interface += indent + tab + "function "
    else:
        f_interface += indent + tab + "subroutine "

    f_interface += f_symbol + "("

    if (is_function):
        initial_indent = len(indent + tab + "function " + f_symbol + "(") * " "
    else:
        initial_indent = len(indent + tab + "subroutine " + f_symbol + "(") * " "

    # loop over the arguments to compose the first line
    for j in range(1,len(function)):
        if (j != 1):
            f_interface += ", "
        if (j%9 == 0):
            f_interface += "&\n" + initial_indent

        f_interface += function[j][2]

    f_interface += ") &\n"
    f_interface += indent + tab + "  " + "bind(c, name='" + c_symbol +"')\n"

    # add common header
    f_interface += indent + 2*tab + "use iso_c_binding\n"
    f_interface += indent + 2*tab + mpi_header + "\n"
    # import derived types
    for derived_type in used_derived_types:
        f_interface += indent + 2*tab + "import " + derived_type +"\n"
    f_interface += indent + 2*tab + "implicit none\n"


    # add the return value of the function
    if (is_function):
        f_interface +=  indent + 2*tab + iso_c_interface_type(function[0], True) + "_c"
        f_interface += "\n"

    # loop over the arguments to describe them
    for j in range(1,len(function)):
        f_interface += indent + 2*tab + iso_c_interface_type(function[j], False)
        f_interface += "\n"

    if (is_function):
        f_interface += indent + tab + "end function\n"
    else:
        f_interface += indent + tab + "end subroutine\n"

    f_interface += indent + "end interface\n"

    return f_interface
# end

#-------------------------------------------------------------------------------
def fortran_wrapper(function):
    """Generate a wrapper for a function.
       void functions in C will be called as subroutines,
       functions in C will be turned to subroutines by appending
       the return value as the last argument."""

    # is it a function or a subroutine
    if (function[0][0] == "void"):
        is_function = False
    else:
        is_function = True

    c_symbol = function[0][2]
    f_symbol = c_symbol + "_c"
    c_symbol = re.sub('_fortran', '', c_symbol)

    if (is_function):
        initial_indent_signature = len(indent + "function " + c_symbol + "(") * " "
        initial_indent_call      = len(indent + tab + "ret_val = " + f_symbol + "(") * " "
    else:
        initial_indent_signature = len(indent + "subroutine " + c_symbol + "(") * " "
        initial_indent_call      = len(indent + tab + "call " + f_symbol + "(") * " "

    # loop over the arguments to compose the first line and call line
    signature_line = ""
    call_line = ""
    double_pointers = []
    for j in range(1,len(function)):
        if (j != 1):
            signature_line += ", "
            call_line += ", "

        # do not make the argument list too long
        if (j%9 == 0):
            call_line      += "&\n" + initial_indent_call
            signature_line += "&\n" + initial_indent_signature

        # pointers
        arg_type    = function[j][0]
        arg_pointer = function[j][1]
        arg_name    = function[j][2]

        signature_line += arg_name
        if (arg_pointer == "**"):
            aux_name = arg_name + "_aux"
            call_line += aux_name
            double_pointers.append(arg_name)
        elif (arg_pointer == "*"):
            call_line += "c_loc(" + arg_name + ")"
        else:
            call_line += arg_name

    contains_derived_types = False
    for arg in function:
        if (arg[0] in derived_types):
            contains_derived_types = True

    # initialize a string with the fortran interface
    f_wrapper = ""
    if (is_function):
        f_wrapper += indent + "function "
    else:
        f_wrapper += indent + "subroutine "
    f_wrapper += c_symbol + "("

    # add the info argument at the end
    f_wrapper += signature_line
    if (is_function):
        if (len(function) > 1):
            f_wrapper += ""

        return_type = function[0][0]
        return_pointer = function[0][1]
        # if (return_type == "int"):
        #    return_var = "info"
        # else:
        return_var = 'ret_val' # return_variables_dict[return_type]

        f_wrapper += ') result(' + return_var

    f_wrapper += ")\n"

    # add common header
    f_wrapper += indent + tab + "use iso_c_binding\n"
    f_wrapper += indent + tab + mpi_header + "\n"
    f_wrapper += indent + tab + "implicit none\n"

    # loop over the arguments to describe them
    for j in range(1,len(function)):
        f_wrapper += indent + tab + iso_c_wrapper_type(function[j]) + "\n"

    # add the return info value of the function
    if (is_function):
        if (function[0][1] == "*"):
            f_target = ", pointer"
        else:
            f_target = ""

        if (return_pointer == "*"):
            # do not export intents
            #f_wrapper += indent + tab + types_dict[return_type] + ", intent(out)" + f_target + " :: " + return_var + "\n"
            f_wrapper += indent + tab + "type(c_ptr) :: " + return_var + "\n"
        else:
            f_wrapper += indent + tab + types_dict[return_type] + f_target + " :: " + return_var + "\n"

    f_wrapper += "\n"

    # loop over potential double pointers and generate auxiliary variables for them
    for double_pointer in double_pointers:
        aux_name = double_pointer + "_aux"
        f_wrapper += indent + tab + "type(c_ptr) :: " + aux_name + "\n"
        f_wrapper += "\n"

    if (is_function):
        f_return = return_var
        f_return += " = "
    else:
        f_return = "call "

    # generate the call to the C function
    # if (is_function and return_pointer == "*"):
    #    f_wrapper += indent + tab + "call c_f_pointer(" + f_symbol + "(" + call_line + "), " + return_var + ")\n"
    #else:
    f_wrapper += indent + tab + f_return + f_symbol + "(" + call_line + ")\n"

    # loop over potential double pointers and translate them to Fortran pointers
    for double_pointer in double_pointers:
        aux_name = double_pointer + "_aux"
        f_wrapper += indent + tab + "call c_f_pointer(" + aux_name + ", " + double_pointer + ")\n"

    if (is_function):
        f_wrapper += indent + "end function\n"
    else:
        f_wrapper += indent + "end subroutine\n"

    return f_wrapper
# end

#-------------------------------------------------------------------------------
def write_module(output, module_name, enum_list, struct_list, function_list):
    """Generate a single Fortran module. Its structure will be:
       enums converted to constants
       structs converted to derived types
       interfaces of all C functions
       Fortran wrappers of the C functions"""

    (dir, file) = os.path.split(output)
    if (not os.path.exists(dir)):
        os.makedirs(dir)

    modulefile = open(output, "w")

    modulefile.write(
'''!>
!>------------------------------------------------------------------------------
!> Copyright (c) 2017-2022, University of Tennessee
!> All rights reserved.
!>
!> Redistribution and use in source and binary forms, with or without
!> modification, are permitted provided that the following conditions are met:
!>     * Redistributions of source code must retain the above copyright
!>       notice, this list of conditions and the following disclaimer.
!>     * Redistributions in binary form must reproduce the above copyright
!>       notice, this list of conditions and the following disclaimer in the
!>       documentation and/or other materials provided with the distribution.
!>     * Neither the name of the University of Tennessee nor the
!>       names of its contributors may be used to endorse or promote products
!>       derived from this software without specific prior written permission.
!>
!> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
!> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
!> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
!> ARE DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
!> DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
!> (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
!> LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
!> ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
!> (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
!> SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
!>------------------------------------------------------------------------------
!> This research was supported by the Exascale Computing Project (17-SC-20-SC),
!> a collaborative effort of two U.S. Department of Energy organizations (Office
!> of Science and the National Nuclear Security Administration) responsible for
!> the planning and preparation of a capable exascale ecosystem, including
!> software, applications, hardware, advanced system engineering and early
!> testbed platforms, in support of the nation's exascale computing imperative.
!>------------------------------------------------------------------------------
!> For assistance with SLATE, email <slate-user@icl.utk.edu>.
!> You can also join the "SLATE User" Google group by going to
!> https:!>groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user,
!> signing in with your Google credentials, and then clicking "Join group".
!>------------------------------------------------------------------------------

!>------------------------------------------------------------------------------
!> Auto-generated file by ''' + sys.argv[0] + '\n'
)
    modulefile.write("module " + module_name + "\n")

    # common header
    modulefile.write(indent + "use iso_c_binding\n")
    modulefile.write(indent + mpi_header + "\n")
    modulefile.write(indent + "implicit none\n\n")

    # enums
    if (len(enum_list) > 0):
        modulefile.write(indent + "! C enums converted to constants.\n")

        for enum in enum_list:
            f_interface = fortran_interface_enum(enum)
            modulefile.write(f_interface + "\n")

    # derived types
    if (len(struct_list) > 0):
        modulefile.write(indent + "! C structs converted to derived types.\n")
        # todo
        for struct in struct_list:
            if ['struct', '', 'slate_Options'] in struct:
                continue
            f_interface = fortran_interface_struct(struct)
            modulefile.write(f_interface + "\n")

    # functions
    if (len(function_list) > 0):
        modulefile.write(indent + "! Interfaces of the C functions.\n")

        for function in function_list:
            f_interface = fortran_interface_function(function)
            modulefile.write(f_interface + "\n")

        modulefile.write("!!" + "-" * 78 + "\n")
        modulefile.write("contains\n\n")

        modulefile.write(indent + "! Wrappers of the C functions.\n")

        for function in function_list:
            f_wrapper = fortran_wrapper(function)
            modulefile.write(f_wrapper + "\n")

    modulefile.write("end module " + module_name + "\n")

    modulefile.close()
# end

#-------------------------------------------------------------------------------
def parse_prototypes(preprocessed_list):
    """Each prototype will be parsed into a list of its arguments."""

    function_list = []
    for proto in preprocessed_list:
        if (proto.find("(") == -1):
            continue

        # extract the part of the function from the prototype
        fun_parts = proto.split("(")
        fun_def   = str.strip(fun_parts[0])

        exclude_this_function = False
        for exclude in exclude_list:
            if (fun_def.find(exclude) != -1):
                exclude_this_function = True

        if (exclude_this_function):
            continue

        # clean keywords
        fun_def = fun_def.replace("^static\s", "")

        # extract arguments from the prototype and make a list from them
        if (len(fun_parts) > 1):
            fun_args = fun_parts[1]
        else:
            fun_args = ""

        fun_args = fun_args.split(")")[0]
        fun_args = fun_args.replace(";", "")
        fun_args = re.sub(r"volatile", "", fun_args)
        fun_args = fun_args.replace("\n", "")
        fun_args_list = fun_args.split(",")

        # generate argument list
        argument_list = []
        fun_def = fun_def.replace("^static\s", "")
        if "struct" in fun_def:
            new_fun_def = fun_def.split(",")
            fun_def = new_fun_def[2]

        # function itself on the first position
        argument_list.append(parse_triple(fun_def))
        # append arguments
        for arg in fun_args_list:
            if (not (arg == "" or arg == " ")):
                arg = arg.replace("const", "")
                argument_list.append(parse_triple(arg))

        # add it only if there is no duplicity with previous one
        is_function_already_present = False
        fun_name = argument_list[0][2]
        for fun in function_list:
            if (fun_name == fun[0][2]):
                is_function_already_present = True

        if (not is_function_already_present):
            function_list.append(argument_list)

    return function_list
# end

#-------------------------------------------------------------------------------
def parse_structs(preprocessed_list):
    """Each struct will be parsed into a list of its arguments."""

    struct_list = []
    for proto in preprocessed_list:
        if "{" not in proto or "}" not in proto:
            continue

        # extract the part of the function from the prototype
        fun_parts = proto.split("{")

        if (fun_parts[0].find("typedef struct") > -1):
            args_string = fun_parts[1]
            parts = args_string.split("}")
            args_string = parts[0].strip()
            args_string = re.sub(r"volatile", "", args_string)

            if (len(parts) > 1):
                name_string = parts[1]
                name_string = re.sub(r"(?m),", "", name_string)
                name_string = name_string.strip()
            else:
                print("Error: Cannot detect name for ", proto)
                name_string = "name_not_recognized"

            args_list = args_string.split(",")
            params = [];
            params.append(["struct","",name_string])
            for arg in args_list:
                if (not (arg == "" or arg == " ")):
                    params.append(parse_triple(arg))

            struct_list.append(params)
            derived_types.append(name_string)

    # reorder the list so that only defined types are exported
    go_again = True
    while (go_again):
        go_again = False
        for istruct in range(0,len(struct_list)-1):
            struct = struct_list[istruct]
            for j in range(1,len(struct)-1):
                type_name = struct_list[istruct][j][0]

                if (type_name in derived_types):

                    # try to find the name in the registered types
                    definedEarlier = False
                    for jstruct in range(0,istruct):
                        struct2 = struct_list[jstruct]
                        that_name = struct2[0][2]
                        if (that_name == type_name):
                            definedEarlier = True

                    # if not found, try to find it behind
                    if (not definedEarlier):
                        definedLater = False
                        for jstruct in range(istruct+1,len(struct_list)-1):
                            struct2 = struct_list[jstruct]
                            that_name = struct2[0][2]
                            if (that_name == type_name):
                                index = jstruct
                                definedLater = True

                        # swap the entries
                        if (definedLater):
                            print("Swapping " + struct_list[istruct][0][2] + " and " + struct_list[index][0][2])
                            tmp = struct_list[index]
                            struct_list[index] = struct_list[istruct]
                            struct_list[istruct] = tmp
                            go_again = True
                            break
                        else:
                            print("Error: Cannot find a derived type " + type_name + " in imported structs.")

            if (go_again):
                break

    return struct_list
# end

#-------------------------------------------------------------------------------
def parse_enums(preprocessed_list):
    """Each enum will be parsed into a list of its arguments."""

    enum_list = []
    for proto in preprocessed_list:
        # extract the part of the function from the prototype
        fun_parts = proto.split("{")

        if (fun_parts[0].find("typedef enum") > -1):
            args_string = fun_parts[1];
            args_string = re.sub(r"}", "", args_string)
            args_list   = args_string.split(",")

            params = [];
            for args in args_list:
                args = args.strip()
                if (args != ""):
                    values = args.split("=")
                    name = values[0].strip()
                    if (len(values) > 1):
                       value = values[1].strip()
                    else:
                        if (len(params) > 0):
                            value = str(int(params[len(params)-1][1]) + 1)
                        else:
                            value = "0"

                    params.append([name, value])

            enum_list.append(params)

    return enum_list
# end

#-------------------------------------------------------------------------------
def preprocess_list(initial_list):
    """Preprocessing and cleaning of the header file.
       Works with a list of strings.
       Produces a new list in which each function, enum or struct
       corresponds to a single item."""

    # merge braces
    list1 = []
    merged_line = ""
    nopen = 0
    inStruct = False
    for line in initial_list:
        if (line.find("struct") > -1):
            inStruct = True

        if (inStruct):
            split_character = ","
        else:
            split_character = ""

        nopen += line.count("{") - line.count("}")
        merged_line += line + split_character

        if (nopen <= 0):
            list1.append(merged_line)
            merged_line = ""
            isOpen   = False
            inStruct = False
            nopen = 0

    # merge structs
    list2 = []
    merged_line = ""
    for line in list1:
        merged_line += line

        if (line.find("struct") == -1):
            list2.append(merged_line)
            merged_line = ""

    # clean orphan braces
    list3 = []
    for line in list2:
        if (line.strip() != "}"):
            list3.append(line)

    return list3
# end

#-------------------------------------------------------------------------------
def polish_file(whole_file):
    """Preprocessing and cleaning of the header file.
       Do not change the order of the regular expressions !
       Works with a long string."""

    clean_file = whole_file

    # Remove C comments:
    clean_file = re.sub(r"(?s)/\*.*?\*/", "", clean_file)
    clean_file = re.sub(r"//.*", "", clean_file)
    # Remove C directives (multilines then monoline):
    clean_file = re.sub(r"(?m)^#(.*[\\][\n])+.*?$", "", clean_file)
    clean_file = re.sub("(?m)^#.*$", "", clean_file)
    clean_file = re.sub("(?m)#.*", "", clean_file)
    # Remove TABs and overnumerous spaces:
    clean_file = clean_file.replace("\t", " ")
    clean_file = re.sub("[ ]{2,}", " ", clean_file)
    # Remove extern C statement:
    clean_file = re.sub("(?m)^(extern).*$", "", clean_file)
    # Remove empty lines:
    clean_file = re.sub(r"(?m)^\n$", "", clean_file)
    # Merge structs
    clean_file = re.sub(r"(?m)$", "", clean_file)
    # Merge string into single line
    clean_file = re.sub(r"\n", "", clean_file)
    # Remove [] and replace them with pointer
    clean_file = re.sub(r"slate_Options opts\[\]", "slate_Options* opts", clean_file)
    # Replace mpi_comm with mpi_comm_ to avoid mpi type conflict
    clean_file = re.sub(r"mpi_comm", "mpi_comm_", clean_file)
    # Split the line based on ";" and "}"
    clean_file = re.sub(r";", "\n", clean_file)
    clean_file = re.sub(r"}", "}\n", clean_file)

    # convert the string to a list of strings
    initial_list_ = clean_file.split("\n")

    initial_list = []
    for list in initial_list_:
        if "_struct_" in list:
            continue
        initial_list.append(list)

    return initial_list
# end

#-------------------------------------------------------------------------------
def main():
    # common cleaned header files
    preprocessed_list = []

    # source header files
    for filename in opts.args:
        # source a header file
        c_header_file = open(filename, 'r').read()

        # clean the string (remove comments, macros, etc.)
        # convert the string to a list of strings
        initial_list = polish_file(c_header_file)

        # process the list so that each enum, struct or function
        # are just one item
        nice_list = preprocess_list(initial_list)

        # compose all files into one big list
        preprocessed_list += nice_list

    # register all enums
    enum_list = parse_enums(preprocessed_list)

    # register all structs
    struct_list = parse_structs(preprocessed_list)

    # register all individual functions and their signatures
    function_list = parse_prototypes(preprocessed_list)

    # export the module
    write_module(
        opts.output, module_name, enum_list, struct_list, function_list)
    print( "Exported file:", opts.output )
# end

# execute the program
main()
