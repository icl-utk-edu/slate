#!/bin/sh
#
# Finds doxygen groups that are in use,  sorts & puts in file "ingroup"
# Finds doxygen groups that are defined, sorts & puts in file "defgroup"
# Doing
#     diff ingroup defgroup
# provides an easy way to see what groups are used vs. defined.

export src=`git ls-files | egrep '\.(hh|cc)'`

egrep -h '@(addto|in)group' ${src} | \
	perl -pe 's#/// +##;  s/^ *\*//;  s/^ +//;  s/\@(addto|in)group/\@group/;' | \
	sort --unique > groups-used.txt

egrep -h '^ *@defgroup' docs/doxygen/groups.dox | \
    egrep -v 'group_|core_blas' | \
    perl -pe 's/^ *\@defgroup +(\w+).*/\@group $1/;' | \
	sort > groups-defined.txt

echo opendiff groups-used.txt groups-defined.txt

