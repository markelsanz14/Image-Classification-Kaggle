FILES="$1/*.jpg"
for f in $FILES
do
	convert $f -resize 256x256\!  "../NewImagesColor/7-"$(basename $f)
done
