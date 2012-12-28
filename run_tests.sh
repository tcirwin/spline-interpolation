#!/bin/bash --
TESTS=tests/*.in


for file in $TESTS
do
   filename="${file%.*}"

   ./splinetest < $filename".in" > $filename".myout"
	echo -------------------- $filename --------------------- 
	diff $filename".out" $filename".myout"
   rm $filename".myout"
done
