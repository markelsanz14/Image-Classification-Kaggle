FILES="$1/*.jpg"
for f in $FILES
do
	convert $f -resize 256x256\! -colorspace Gray "../NewImagesTest/"$(basename $f)
done
