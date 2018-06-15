/*
    ANNT - Artificial Neural Networks C++ library

    MNIST handwritten digits data set parser

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

#include "MNISTParser.hpp"

using namespace std;

#define MNIST_LABELS_FILE_MAGIC    (0x00000801)
#define MNIST_IMAGES_FILE_MAGIC    (0x00000803)
#define LABELS_READING_BUFFER_SIZE (1000)

namespace ANNT {

static uint32_t ReverseEndian32( uint32_t v )
{
    return ( v >> 24 ) | ( ( v & 0xFF0000 ) >> 8 ) | ( ( v & 0xFF00 ) << 8 ) | ( ( v & 0xFF ) << 24 );
}

// Loads labels from the specified MNIST labels' file
bool MNISTParser::LoadLabels( const std::string& fileName, uvector_t& labels )
{
    FILE* file = fopen( fileName.c_str( ), "rb" );
    bool  ret  = false;

    if ( file != nullptr )
    {
        uint32_t magic = 0, labelsCount = 0;

        if ( ( fread( &magic, 4, 1, file ) == 1 ) &&
             ( fread( &labelsCount, 4, 1, file ) == 1 ) )
        {
            magic       = ReverseEndian32( magic );
            labelsCount = ReverseEndian32( labelsCount );

            if ( magic == MNIST_LABELS_FILE_MAGIC )
            {
                uint8_t buffer[LABELS_READING_BUFFER_SIZE];
                size_t  toLoad = labelsCount;
                size_t  loaded = 0;
                size_t  read   = 0;

                labels = uvector_t( toLoad );

                while ( loaded != toLoad )
                {
                    size_t nextRead = ( toLoad - loaded );

                    if ( nextRead > LABELS_READING_BUFFER_SIZE )
                    {
                        nextRead = LABELS_READING_BUFFER_SIZE;
                    }

                    read = fread( buffer, 1, nextRead, file );

                    if ( read == nextRead )
                    {
                        for ( size_t i = 0; i < read; i++ )
                        {
                            labels[loaded + i] = static_cast<size_t>( buffer[i] );
                        }

                        loaded += read;
                    }
                }

                ret = true;
            }
        }

        fclose( file );
    }

    return ret;
}

// Loads images from the specified MNIST images' file
bool MNISTParser::LoadImages( const string& fileName, vector<fvector_t>& images,
                              float_t scaleMin, float_t scaleMax, size_t xPad, size_t yPad )
{
    FILE* file = fopen( fileName.c_str( ), "rb" );
    bool  ret  = false;

    if ( file != nullptr )
    {
        uint32_t magic = 0, imageCount = 0, width = 0, height = 0;

        if ( ( fread( &magic, 4, 1, file ) == 1 ) &&
             ( fread( &imageCount, 4, 1, file ) == 1 ) &&
             ( fread( &height, 4, 1, file ) == 1 ) &&
             ( fread( &width, 4, 1, file ) == 1 ) )
        {
            magic      = ReverseEndian32( magic );
            imageCount = ReverseEndian32( imageCount );
            width      = ReverseEndian32( width );
            height     = ReverseEndian32( height );

            if ( magic == MNIST_IMAGES_FILE_MAGIC )
            {
                size_t   imageSize    = width * height;
                uint8_t* buffer       = new uint8_t[imageSize];
                size_t   toLoad       = imageCount;
                size_t   read         = 0;

                uint32_t paddedWidth  = width  + xPad * 2;
                uint32_t paddedHeight = height + yPad * 2;
                uint32_t paddedSize   = paddedWidth * paddedHeight;

                for ( size_t i = 0; i < toLoad; i++ )
                {
                    if ( ( read = fread( buffer, 1, imageSize, file ) ) == imageSize )
                    {
                        fvector_t image( paddedSize, scaleMin );

                        for ( uint32_t y = 0, j = 0; y < height; y++ )
                        {
                            for ( uint32_t x = 0; x < width; x++, j++ )
                            {
                                image[( y + yPad ) * paddedWidth + xPad + x] = ( static_cast<float_t>( buffer[j] ) / 255 ) *
                                                                               ( scaleMax - scaleMin ) + scaleMin;
                            }
                        }

                        images.push_back( image );
                    }
                }

                delete [] buffer;

                ret = true;
            }
        }

        fclose( file );
    }

    return ret;
}

} // namespace ANNT
