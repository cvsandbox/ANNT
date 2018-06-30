/*
    ANNT - Artificial Neural Networks C++ library

    CIFAR-10 data set parser

    Copyright (C) 2018, cvsandbox, cvsandbox@gmail.com

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
*/

#include <stdio.h>
#include <stdint.h>

#include "CIFARParser.hpp"

using namespace std;

namespace ANNT {

#define CIFAR_IMAGE_WIDTH   (32)
#define CIFAR_IMAGE_HEIGHT  (32)
#define CIFAR_IMAGE_PLANES  (3)
#define CIFAR_IMAGE_IN_FILE (10000)

// Loads images and labels from the specified CIFAR-10 dataset file (appends to the provided vectors)
bool CIFARParser::LoadDataSet( const std::string& fileName, uvector_t& labels, std::vector<fvector_t>& images,
                               float_t scaleMin, float_t scaleMax  )
{
    FILE* file = fopen( fileName.c_str( ), "rb" );
    bool  ret  = false;

    if ( file != nullptr )
    {
        size_t   imageSize  = CIFAR_IMAGE_WIDTH * CIFAR_IMAGE_HEIGHT * CIFAR_IMAGE_PLANES;
        size_t   sampleSize = imageSize + 1; // extra byte for label
        uint8_t* buffer     = new uint8_t[sampleSize];
        size_t   read;

        for ( size_t imageIndex = 0; imageIndex < CIFAR_IMAGE_IN_FILE; imageIndex++ )
        {
            read = fread( buffer, 1, sampleSize, file );

            if ( read == sampleSize )
            {
                fvector_t image( imageSize, scaleMin );

                for ( size_t depth = 0, j = 0; depth < CIFAR_IMAGE_PLANES; depth++ )
                {
                    for ( size_t y = 0; y < CIFAR_IMAGE_HEIGHT; y++ )
                    {
                        for ( size_t x = 0; x < CIFAR_IMAGE_WIDTH; x++, j++ )
                        {
                            image[j] = ( static_cast< float_t >( buffer[j + 1] ) / 255 ) * ( scaleMax - scaleMin ) + scaleMin;
                        }
                    }
                }

                labels.push_back( buffer[0] );
                images.push_back( image );
            }
        }

        delete [] buffer;
        fclose( file );
        ret = true;
    }

    return ret;
}

} // namespace ANNT
