{\rtf1\ansi\ansicpg1252\cocoartf2869
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 Menlo-Regular;}
{\colortbl;\red255\green255\blue255;\red194\green11\blue35;\red255\green255\blue255;\red24\green24\blue24;
\red11\green34\blue86;}
{\*\expandedcolortbl;;\cssrgb\c81176\c13333\c18039;\cssrgb\c100000\c100000\c100000;\cssrgb\c12549\c12549\c12549;
\cssrgb\c3922\c18824\c41176;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs24 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 FROM\cf4 \strokec4  python:3.11-slim\cb1 \
\
\cf2 \cb3 \strokec2 WORKDIR\cf4 \strokec4  /app\cb1 \
\
\cf2 \cb3 \strokec2 RUN\cf4 \strokec4  apt-get update && apt-get install -y \\\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     libgl1 \\\cb1 \
\cb3     libglib2.0-0 \\\cb1 \
\cb3     && rm -rf /var/lib/apt/lists/*\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 COPY\cf4 \strokec4  requirements.txt .\cb1 \
\cf2 \cb3 \strokec2 RUN\cf4 \strokec4  pip install --no-cache-dir -r requirements.txt\cb1 \
\
\cf2 \cb3 \strokec2 COPY\cf4 \strokec4  . .\cb1 \
\
\cf2 \cb3 \strokec2 CMD\cf4 \strokec4  [\cf5 \strokec5 "uvicorn"\cf4 \strokec4 , \cf5 \strokec5 "main:app"\cf4 \strokec4 , \cf5 \strokec5 "--host"\cf4 \strokec4 , \cf5 \strokec5 "0.0.0.0"\cf4 \strokec4 , \cf5 \strokec5 "--port"\cf4 \strokec4 , \cf5 \strokec5 "8080"\cf4 \strokec4 ]\cb1 \
}
