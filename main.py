{\rtf1\ansi\ansicpg1252\cocoartf2869
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 Menlo-Regular;}
{\colortbl;\red255\green255\blue255;\red157\green0\blue210;\red255\green255\blue255;\red24\green24\blue24;
\red10\green58\blue158;\red0\green0\blue0;\red11\green34\blue86;\red129\green40\blue2;\red0\green0\blue255;
\red109\green51\blue215;\red194\green11\blue35;\red24\green26\blue30;\red32\green108\blue135;\red101\green76\blue29;
\red19\green118\blue70;\red91\green100\blue110;}
{\*\expandedcolortbl;;\cssrgb\c68627\c0\c85882;\cssrgb\c100000\c100000\c100000;\cssrgb\c12549\c12549\c12549;
\cssrgb\c1961\c31373\c68235;\cssrgb\c0\c0\c0;\cssrgb\c3922\c18824\c41176;\cssrgb\c58431\c21961\c0;\cssrgb\c0\c0\c100000;
\cssrgb\c50980\c31373\c87451;\cssrgb\c81176\c13333\c18039;\cssrgb\c12157\c13725\c15686;\cssrgb\c14902\c49804\c60000;\cssrgb\c47451\c36863\c14902;
\cssrgb\c3529\c52549\c34510;\cssrgb\c43137\c46667\c50588;}
\margl1440\margr1440\vieww29200\viewh15300\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs24 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 from\cf4 \strokec4  fastapi \cf2 \strokec2 import\cf4 \strokec4  FastAPI, UploadFile, File\cb1 \
\cf2 \cb3 \strokec2 from\cf4 \strokec4  pathlib \cf2 \strokec2 import\cf4 \strokec4  Path\cb1 \
\cf2 \cb3 \strokec2 import\cf4 \strokec4  fitz\cb1 \
\cf2 \cb3 \strokec2 import\cf4 \strokec4  cv2\cb1 \
\cf2 \cb3 \strokec2 import\cf4 \strokec4  numpy \cf2 \strokec2 as\cf4 \strokec4  np\cb1 \
\cf2 \cb3 \strokec2 from\cf4 \strokec4  \cf5 \strokec5 PIL\cf4 \strokec4  \cf2 \strokec2 import\cf4 \strokec4  Image\cb1 \
\cf2 \cb3 \strokec2 import\cf4 \strokec4  imagehash\cb1 \
\cf2 \cb3 \strokec2 import\cf4 \strokec4  uuid\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 app \strokec6 =\strokec4  FastAPI()\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 OUTPUT_DIR\cf4 \strokec4  \strokec6 =\strokec4  Path(\cf7 \strokec7 "output"\cf4 \strokec4 )\cb1 \
\cf5 \cb3 \strokec5 OUTPUT_DIR\cf4 \strokec4 .mkdir(\cf8 \strokec8 exist_ok\cf4 \strokec6 =\cf9 \strokec9 True\cf4 \strokec4 )\cb1 \
\
\
\pard\pardeftab720\partightenfactor0
\cf10 \cb3 \strokec10 @app.get\cf4 \strokec4 (\cf7 \strokec7 "/"\cf4 \strokec4 )\cb1 \
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 def\cf4 \strokec4  \cf10 \strokec10 home\cf4 \strokec4 ():\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     \cf2 \strokec2 return\cf4 \strokec4  \{\cf7 \strokec7 "status"\cf4 \strokec4 : \cf7 \strokec7 "Artwork extractor is running"\cf4 \strokec4 \}\cb1 \
\
\
\pard\pardeftab720\partightenfactor0
\cf10 \cb3 \strokec10 @app.post\cf4 \strokec4 (\cf7 \strokec7 "/extract-artwork"\cf4 \strokec4 )\cb1 \
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 async\cf4 \strokec4  \cf11 \strokec11 def\cf4 \strokec4  \cf10 \strokec10 extract_artwork\cf4 \strokec4 (\cf12 \strokec12 file\cf4 \strokec4 : UploadFile \strokec6 =\strokec4  File(\cf5 \strokec5 ...\cf4 \strokec4 )):\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     job_id \strokec6 =\strokec4  \cf13 \strokec13 str\cf4 \strokec4 (uuid.uuid4())\cb1 \
\cb3     job_dir \strokec6 =\strokec4  \cf5 \strokec5 OUTPUT_DIR\cf4 \strokec4  \strokec6 /\strokec4  job_id\cb1 \
\cb3     job_dir.mkdir(\cf8 \strokec8 parents\cf4 \strokec6 =\cf9 \strokec9 True\cf4 \strokec4 , \cf8 \strokec8 exist_ok\cf4 \strokec6 =\cf9 \strokec9 True\cf4 \strokec4 )\cb1 \
\
\cb3     pdf_path \strokec6 =\strokec4  job_dir \strokec6 /\strokec4  \strokec6 file\strokec4 .filename\cb1 \
\
\cb3     \cf2 \strokec2 with\cf4 \strokec4  \cf14 \strokec14 open\cf4 \strokec4 (pdf_path, \cf7 \strokec7 "wb"\cf4 \strokec4 ) \cf2 \strokec2 as\cf4 \strokec4  f:\cb1 \
\cb3         f.write(\cf2 \strokec2 await\cf4 \strokec4  \strokec6 file\strokec4 .read())\cb1 \
\
\cb3     extracted \strokec6 =\strokec4  process_pdf(pdf_path, job_dir)\cb1 \
\
\cb3     \cf2 \strokec2 return\cf4 \strokec4  \{\cb1 \
\cb3         \cf7 \strokec7 "job_id"\cf4 \strokec4 : job_id,\cb1 \
\cb3         \cf7 \strokec7 "artworks_found"\cf4 \strokec4 : \cf14 \strokec14 len\cf4 \strokec4 (extracted),\cb1 \
\cb3         \cf7 \strokec7 "artworks"\cf4 \strokec4 : extracted\cb1 \
\cb3     \}\cb1 \
\
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 def\cf4 \strokec4  \cf10 \strokec10 process_pdf\cf4 \strokec4 (\cf12 \strokec12 pdf_path\cf4 \strokec4 : Path, \cf12 \strokec12 job_dir\cf4 \strokec4 : Path):\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     doc \strokec6 =\strokec4  fitz.open(pdf_path)\cb1 \
\cb3     results \strokec6 =\strokec4  []\cb1 \
\
\cb3     \cf2 \strokec2 for\cf4 \strokec4  page_index \cf2 \strokec2 in\cf4 \strokec4  \cf14 \strokec14 range\cf4 \strokec4 (\cf14 \strokec14 len\cf4 \strokec4 (doc)):\cb1 \
\cb3         page \strokec6 =\strokec4  doc[page_index]\cb1 \
\
\cb3         pix \strokec6 =\strokec4  page.get_pixmap(\cf8 \strokec8 matrix\cf4 \strokec6 =\strokec4 fitz.Matrix(\cf15 \strokec15 2\cf4 \strokec4 , \cf15 \strokec15 2\cf4 \strokec4 ))\cb1 \
\cb3         page_img_path \strokec6 =\strokec4  job_dir \strokec6 /\strokec4  \cf11 \strokec11 f\cf7 \strokec7 "page_\cf11 \strokec11 \{\cf4 \strokec4 page_index \strokec6 +\strokec4  \cf15 \strokec15 1\cf11 \strokec11 \}\cf7 \strokec7 .png"\cf4 \cb1 \strokec4 \
\cb3         pix.save(page_img_path)\cb1 \
\
\cb3         crops \strokec6 =\strokec4  extract_artwork_panels(page_img_path, job_dir, page_index \strokec6 +\strokec4  \cf15 \strokec15 1\cf4 \strokec4 )\cb1 \
\cb3         results.extend(crops)\cb1 \
\
\cb3     \cf2 \strokec2 return\cf4 \strokec4  results\cb1 \
\
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 def\cf4 \strokec4  \cf10 \strokec10 extract_artwork_panels\cf4 \strokec4 (\cf12 \strokec12 page_img_path\cf4 \strokec4 : Path, \cf12 \strokec12 job_dir\cf4 \strokec4 : Path, \cf12 \strokec12 page_number\cf4 \strokec4 : \cf13 \strokec13 int\cf4 \strokec4 ):\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     img \strokec6 =\strokec4  cv2.imread(\cf13 \strokec13 str\cf4 \strokec4 (page_img_path))\cb1 \
\cb3     height, width \strokec6 =\strokec4  img.shape[:\cf15 \strokec15 2\cf4 \strokec4 ]\cb1 \
\
\cb3     \cf16 \strokec16 # The artwork panels are usually below the garment mockup.\cf4 \cb1 \strokec4 \
\cb3     \cf16 \strokec16 # Start by looking in the lower half of the page.\cf4 \cb1 \strokec4 \
\cb3     lower_half \strokec6 =\strokec4  img[\cf13 \strokec13 int\cf4 \strokec4 (height \strokec6 *\strokec4  \cf15 \strokec15 0.45\cf4 \strokec4 ):height, :]\cb1 \
\
\cb3     gray \strokec6 =\strokec4  cv2.cvtColor(lower_half, cv2.\cf5 \strokec5 COLOR_BGR2GRAY\cf4 \strokec4 )\cb1 \
\
\cb3     \cf16 \strokec16 # Find large non-black / non-white rectangular areas.\cf4 \cb1 \strokec4 \
\cb3     _, thresh \strokec6 =\strokec4  cv2.threshold(gray, \cf15 \strokec15 30\cf4 \strokec4 , \cf15 \strokec15 255\cf4 \strokec4 , cv2.\cf5 \strokec5 THRESH_BINARY\cf4 \strokec4 )\cb1 \
\
\cb3     contours, _ \strokec6 =\strokec4  cv2.findContours(\cb1 \
\cb3         thresh,\cb1 \
\cb3         cv2.\cf5 \strokec5 RETR_EXTERNAL\cf4 \strokec4 ,\cb1 \
\cb3         cv2.\cf5 \strokec5 CHAIN_APPROX_SIMPLE\cf4 \cb1 \strokec4 \
\cb3     )\cb1 \
\
\cb3     results \strokec6 =\strokec4  []\cb1 \
\
\cb3     \cf2 \strokec2 for\cf4 \strokec4  i, contour \cf2 \strokec2 in\cf4 \strokec4  \cf14 \strokec14 enumerate\cf4 \strokec4 (contours):\cb1 \
\cb3         x, y, w, h \strokec6 =\strokec4  cv2.boundingRect(contour)\cb1 \
\
\cb3         area \strokec6 =\strokec4  w \strokec6 *\strokec4  h\cb1 \
\cb3         \cf2 \strokec2 if\cf4 \strokec4  area \strokec6 <\strokec4  \cf15 \strokec15 50000\cf4 \strokec4 :\cb1 \
\cb3             \cf2 \strokec2 continue\cf4 \cb1 \strokec4 \
\
\cb3         \cf16 \strokec16 # Avoid tiny color circles and text areas\cf4 \cb1 \strokec4 \
\cb3         \cf2 \strokec2 if\cf4 \strokec4  h \strokec6 <\strokec4  \cf15 \strokec15 150\cf4 \strokec4  \cf9 \strokec9 or\cf4 \strokec4  w \strokec6 <\strokec4  \cf15 \strokec15 150\cf4 \strokec4 :\cb1 \
\cb3             \cf2 \strokec2 continue\cf4 \cb1 \strokec4 \
\
\cb3         y_absolute \strokec6 =\strokec4  y \strokec6 +\strokec4  \cf13 \strokec13 int\cf4 \strokec4 (height \strokec6 *\strokec4  \cf15 \strokec15 0.45\cf4 \strokec4 )\cb1 \
\
\cb3         crop \strokec6 =\strokec4  img[y_absolute:y_absolute\strokec6 +\strokec4 h, x:x\strokec6 +\strokec4 w]\cb1 \
\
\cb3         \cf2 \strokec2 if\cf4 \strokec4  is_blank_crop(crop):\cb1 \
\cb3             \cf2 \strokec2 continue\cf4 \cb1 \strokec4 \
\
\cb3         crop_path \strokec6 =\strokec4  job_dir \strokec6 /\strokec4  \cf11 \strokec11 f\cf7 \strokec7 "artwork_page_\cf11 \strokec11 \{\cf4 \strokec4 page_number\cf11 \strokec11 \}\cf7 \strokec7 _\cf11 \strokec11 \{\cf4 \strokec4 i \strokec6 +\strokec4  \cf15 \strokec15 1\cf11 \strokec11 \}\cf7 \strokec7 .png"\cf4 \cb1 \strokec4 \
\cb3         cv2.imwrite(\cf13 \strokec13 str\cf4 \strokec4 (crop_path), crop)\cb1 \
\
\cb3         pil_img \strokec6 =\strokec4  Image.open(crop_path)\cb1 \
\cb3         hash_value \strokec6 =\strokec4  \cf13 \strokec13 str\cf4 \strokec4 (imagehash.phash(pil_img))\cb1 \
\
\cb3         results.append(\{\cb1 \
\cb3             \cf7 \strokec7 "page"\cf4 \strokec4 : page_number,\cb1 \
\cb3             \cf7 \strokec7 "file"\cf4 \strokec4 : \cf13 \strokec13 str\cf4 \strokec4 (crop_path),\cb1 \
\cb3             \cf7 \strokec7 "hash"\cf4 \strokec4 : hash_value,\cb1 \
\cb3             \cf7 \strokec7 "width"\cf4 \strokec4 : w,\cb1 \
\cb3             \cf7 \strokec7 "height"\cf4 \strokec4 : h\cb1 \
\cb3         \})\cb1 \
\
\cb3     \cf2 \strokec2 return\cf4 \strokec4  results\cb1 \
\
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 def\cf4 \strokec4  \cf10 \strokec10 is_blank_crop\cf4 \strokec4 (\cf12 \strokec12 crop\cf4 \strokec4 ):\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     gray \strokec6 =\strokec4  cv2.cvtColor(crop, cv2.\cf5 \strokec5 COLOR_BGR2GRAY\cf4 \strokec4 )\cb1 \
\
\cb3     \cf16 \strokec16 # If very low contrast, likely blank artwork panel\cf4 \cb1 \strokec4 \
\cb3     contrast \strokec6 =\strokec4  gray.std()\cb1 \
\
\cb3     \cf2 \strokec2 return\cf4 \strokec4  contrast \strokec6 <\strokec4  \cf15 \strokec15 8\cf4 \cb1 \strokec4 \
}