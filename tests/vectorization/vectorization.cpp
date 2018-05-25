/*
    ANNT - Artificial Neural Networks C++ library

    AVX/SSE vectorization test

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
#include <string.h>
#include <vector>
#include <chrono>

#include "Types/XAlignedAllocator.hpp"
#include "Tools/XAvxVectorTools.hpp"
#include "Tools/XSseVectorTools.hpp"
#include "Tools/XVectorTools.hpp"

using namespace std;
using namespace std::chrono;
using namespace ANNT;

// Number of tests to run and calculate average performance for
uint32_t TESTS_COUNT      = 10;
// Number of iterations to run for each test
uint32_t ITERATIONS_COUNT = 15;
// Vector size used in all tests
uint32_t VECTOR_SIZE      = 10 * 1000* 1000 + 7;

// 32-byte aligned vector types to make AVX instructions happy
typedef vector<float,  XAlignedAllocator<float,  32>> float_vec_t;
typedef vector<double, XAlignedAllocator<double, 32>> double_vec_t;

// Forward declaration of tests to run
template <typename vecType> float AddTest( const IVectorTools* vectorTools );
template <typename vecType> float MulTest( const IVectorTools* vectorTools );
template <typename vecType> float DotTest( const IVectorTools* vectorTools );
template <typename vecType> float MaxTest( const IVectorTools* vectorTools );

// Parse command line parameters to override defaults
static void ParseCommandLine( int argc, char** argv )
{
    for ( int i = 1; i < argc; i++ )
    {
        bool   parsed   = false;
        size_t paramLen = strlen( argv[i] );

        if ( paramLen >= 2 )
        {
            char* paramStart = &( argv[i][1] );

            if ( ( argv[i][0] == '-' ) || ( argv[i][0] == '/' ) )
            {
                if ( paramLen > 3 )
                {
                    if ( strstr( paramStart, "t:" ) == paramStart )
                    {
                        if ( sscanf( &( argv[i][3] ), "%u", &TESTS_COUNT ) == 1 )
                        {
                            parsed = true;
                        }
                    }
                    else if ( strstr( paramStart, "i:" ) == paramStart )
                    {
                        if ( sscanf( &( argv[i][3] ), "%u", &ITERATIONS_COUNT ) == 1 )
                        {
                            parsed = true;
                        }
                    }
                    else if ( strstr( paramStart, "v:" ) == paramStart )
                    {
                        if ( sscanf( &( argv[i][3] ), "%u", &VECTOR_SIZE ) == 1 )
                        {
                            if ( VECTOR_SIZE < 100 )
                            {
                                VECTOR_SIZE = 100;
                            }
                            parsed = true;
                        }
                    }
                }
            }
        }

        if ( !parsed )
        {
            printf( "Failed parsing parameter or don't know about it: %s \n", argv[i] );
        }
    }

    printf( "Test runs        : %u \n", TESTS_COUNT );
    printf( "Iterations count : %u \n", ITERATIONS_COUNT );
    printf( "Vector size      : %d \n", VECTOR_SIZE );
    printf( "\n" );
}

int main( int argc, char** argv )
{
    printf( "Vectorization test \n" );
    printf( "================== \n" );

    XAvxVectorTools avxVectorTools;
    XSseVectorTools sseVectorTools;
    XVectorTools    defVectorTools;

    bool  avxSupported = false;
    bool  sseSupported = false;

    float avxAddS = 0, avxMulS = 0, avxDotS = 0, avxMaxS = 0;
    float sseAddS = 0, sseMulS = 0, sseDotS = 0, sseMaxS = 0;
    float defAddS = 0, defMulS = 0, defDotS = 0, defMaxS = 0;
    float avxAddD = 0, avxMulD = 0, avxDotD = 0, avxMaxD = 0;
    float sseAddD = 0, sseMulD = 0, sseDotD = 0, sseMaxD = 0;
    float defAddD = 0, defMulD = 0, defDotD = 0, defMaxD = 0;

    ParseCommandLine( argc, argv );

    if ( avxVectorTools.IsAvailable( ) )
    {
        avxSupported = true;
        printf( "AVX tools are available \n" );
    }
    else
    {
        printf( "AVX tools are NOT available \n" );
    }

    if ( sseVectorTools.IsAvailable( ) )
    {
        sseSupported = true;
        printf( "SSE tools are available \n" );
    }
    else
    {
        printf( "SSE tools are NOT available \n" );
    }

    printf( "\n" );
    printf( "Running single precision tests ... \n" );

    if ( avxSupported )
    {
        srand( 0 );
        printf( "\nAVX ADD\n" );
        avxAddS = AddTest<float_vec_t>( &avxVectorTools );
        printf( "\nAVX MUL\n" );
        avxMulS = MulTest<float_vec_t>( &avxVectorTools );
        printf( "\nAVX DOT\n" );
        avxDotS = DotTest<float_vec_t>( &avxVectorTools );
        printf( "\nAVX MAX\n" );
        avxMaxS = MaxTest<float_vec_t>( &avxVectorTools );
    }

    if ( sseSupported )
    {
        srand( 0 );
        printf( "\nSSE ADD\n" );
        sseAddS = AddTest<float_vec_t>( &sseVectorTools );
        printf( "\nSSE MUL\n" );
        sseMulS = MulTest<float_vec_t>( &sseVectorTools );
        printf( "\nSSE DOT\n" );
        sseDotS = DotTest<float_vec_t>( &sseVectorTools );
        printf( "\nSSE MAX\n" );
        sseMaxS = MaxTest<float_vec_t>( &sseVectorTools );
    }

    srand( 0 );
    printf( "\nDEF ADD\n" );
    defAddS = AddTest<float_vec_t>( &defVectorTools );
    printf( "\nDEF MUL\n" );
    defMulS = MulTest<float_vec_t>( &defVectorTools );
    printf( "\nDEF DOT\n" );
    defDotS = DotTest<float_vec_t>( &defVectorTools );
    printf( "\nDEF MAX\n" );
    defMaxS = MaxTest<float_vec_t>( &defVectorTools );

    printf( "\n" );
    printf( "Running double precision tests ... \n" );

    if ( avxSupported )
    {
        srand( 0 );
        printf( "\nAVX ADD\n" );
        avxAddD = AddTest<double_vec_t>( &avxVectorTools );
        printf( "\nAVX MUL\n" );
        avxMulD = MulTest<double_vec_t>( &avxVectorTools );
        printf( "\nAVX DOT\n" );
        avxDotD = DotTest<double_vec_t>( &avxVectorTools );
        printf( "\nAVX MAX\n" );
        avxMaxD = MaxTest<double_vec_t>( &avxVectorTools );
    }

    if ( sseSupported )
    {
        srand( 0 );
        printf( "\nSSE ADD\n" );
        sseAddD = AddTest<double_vec_t>( &sseVectorTools );
        printf( "\nSSE MUL\n" );
        sseMulD = MulTest<double_vec_t>( &sseVectorTools );
        printf( "\nSSE DOT\n" );
        sseDotD = DotTest<double_vec_t>( &sseVectorTools );
        printf( "\nSSE MAX\n" );
        sseMaxD = MaxTest<double_vec_t>( &sseVectorTools );
    }

    srand( 0 );
    printf( "\nDEF ADD\n" );
    defAddD = AddTest<double_vec_t>( &defVectorTools );
    printf( "\nDEF MUL\n" );
    defMulD = MulTest<double_vec_t>( &defVectorTools );
    printf( "\nDEF DOT\n" );
    defDotD = DotTest<double_vec_t>( &defVectorTools );
    printf( "\nDEF MAX\n" );
    defMaxD = MaxTest<double_vec_t>( &defVectorTools );

    printf( "\n\n" );
    printf( "Single precision:\n\n" );
    printf( "\t   Add \t | Mul \t | Dot \t | Max \n" );
    if ( avxSupported )
    {
        printf( "AVX \t | %0.2f | %0.2f | %0.2f | %0.2f \n", avxAddS, avxMulS, avxDotS, avxMaxS );
    }
    if ( sseSupported )
    {
        printf( "SSE \t | %0.2f | %0.2f | %0.2f | %0.2f \n", sseAddS, sseMulS, sseDotS, sseMaxS );
    }
    printf( "DEF \t | %0.2f | %0.2f | %0.2f | %0.2f \n", defAddS, defMulS, defDotS, defMaxS );
    printf( "\n" );

    printf( "Double precision:\n\n" );
    printf( "\t   Add \t | Mul \t | Dot \t | Max \n" );
    if ( avxSupported )
    {
        printf( "AVX \t | %0.2f | %0.2f | %0.2f | %0.2f \n", avxAddD, avxMulD, avxDotD, avxMaxD );
    }
    if ( sseSupported )
    {
        printf( "SSE \t | %0.2f | %0.2f | %0.2f | %0.2f \n", sseAddD, sseMulD, sseDotD, sseMaxD );
    }
    printf( "DEF \t | %0.2f | %0.2f | %0.2f | %0.2f \n", defAddD, defMulD, defDotD, defMaxD );
    printf( "\n" );

	return 0;
}

// Adding elements of two vectors : dst[i] += src[i]
template <typename vecType> float AddTest( const IVectorTools* vectorTools )
{
    vecType src( VECTOR_SIZE );
    vecType dst( VECTOR_SIZE );
    float   avgTime = 0.0f;

    for ( size_t t = 0; t < TESTS_COUNT; t++ )
    {
        for ( size_t i = 0; i < VECTOR_SIZE; i++ )
        {
            src[i] = ( static_cast<float>( rand( ) ) / RAND_MAX ) * float( 2 ) - 1.0f;
            dst[i] = ( static_cast<float>( rand( ) ) / RAND_MAX ) * float( 2 ) - 1.0f;
        }

        steady_clock::time_point start = steady_clock::now( );

        for ( size_t i = 0; i < ITERATIONS_COUNT; i++ )
        {
            vectorTools->Add( src.data( ), dst.data( ), src.size( ) );
        }

        auto timeTaken = duration_cast<std::chrono::milliseconds>( steady_clock::now( ) - start ).count( );

        printf( "time taken: %u \n", static_cast<uint32_t>( timeTaken ) );
        for ( size_t i = 0; i < 8; i++ )
        {
            printf( "%f ", static_cast<float>( dst[i] ) );
        }
        printf( "\n" );
        for ( size_t i = 0; i < 8; i++ )
        {
            printf( "%f ", static_cast<float>( dst[VECTOR_SIZE - 8 + i] ) );
        }
        printf( "\n" );

        avgTime += static_cast<float>( timeTaken );
    }

    avgTime /= TESTS_COUNT;

    return avgTime;
}

// Element wise multiplication of two vectors : dst[i] *= src[i]
template <typename vecType> float MulTest( const IVectorTools* vectorTools )
{
    vecType src( VECTOR_SIZE );
    vecType dst( VECTOR_SIZE );
    float   avgTime = 0.0f;

    for ( size_t t = 0; t < TESTS_COUNT; t++ )
    {
        for ( size_t i = 0; i < VECTOR_SIZE; i++ )
        {
            src[i] = ( static_cast<float>( rand( ) ) / RAND_MAX ) * float( 2 ) - 1.0f;
            dst[i] = ( static_cast<float>( rand( ) ) / RAND_MAX ) * float( 2 ) - 1.0f;
        }

        steady_clock::time_point start = steady_clock::now( );

        for ( size_t i = 0; i < ITERATIONS_COUNT; i++ )
        {
            vectorTools->Mul( src.data( ), dst.data( ), src.size( ) );
        }

        auto timeTaken = duration_cast<std::chrono::milliseconds>( steady_clock::now( ) - start ).count( );

        printf( "time taken: %u \n", static_cast<uint32_t>( timeTaken ) );
        for ( size_t i = 0; i < 8; i++ )
        {
            printf( "%f ", static_cast<float>( dst[i] ) );
        }
        printf( "\n" );
        for ( size_t i = 0; i < 8; i++ )
        {
            printf( "%f ", static_cast<float>( dst[VECTOR_SIZE - 8 + i] ) );
        }
        printf( "\n" );

        avgTime += static_cast<float>( timeTaken );
    }

    avgTime /= TESTS_COUNT;

    return avgTime;
}

// Dot product of two vectors : sum( vec1, vec2 )
template <typename vecType> float DotTest( const IVectorTools* vectorTools )
{
    vecType vec1( VECTOR_SIZE );
    vecType vec2( VECTOR_SIZE );
    float   avgTime = 0.0f;

    for ( size_t t = 0; t < TESTS_COUNT; t++ )
    {
        typename vecType::value_type dot = 0;

        for ( size_t i = 0; i < VECTOR_SIZE; i++ )
        {
            vec1[i] = ( static_cast<float>( rand( ) ) / RAND_MAX ) * float( 2 ) - 1.0f;
            vec2[i] = ( static_cast<float>( rand( ) ) / RAND_MAX ) * float( 2 ) - 1.0f;
        }

        steady_clock::time_point start = steady_clock::now( );

        for ( size_t i = 0; i < ITERATIONS_COUNT; i++ )
        {
            dot = vectorTools->Dot( vec1.data( ), vec2.data( ), vec1.size( ) );
        }

        auto timeTaken = duration_cast<std::chrono::milliseconds>( steady_clock::now( ) - start ).count( );

        printf( "time taken: %u \n", static_cast<uint32_t>( timeTaken ) );
        printf( "dot: %f \n", static_cast<float>( dot ) );

        avgTime += static_cast<float>( timeTaken );
    }

    avgTime /= TESTS_COUNT;

    return avgTime;
}

// Maximum value of vector's elements and the specified alpha value
template <typename vecType> float MaxTest( const IVectorTools* vectorTools )
{
    vecType src( VECTOR_SIZE );
    vecType dst( VECTOR_SIZE );
    float   avgTime = 0.0f;

    typename vecType::value_type alpha = 0;

    for ( size_t t = 0; t < TESTS_COUNT; t++ )
    {
        for ( size_t i = 0; i < VECTOR_SIZE; i++ )
        {
            src[i] = ( static_cast<float>( rand( ) ) / RAND_MAX ) * float( 2 ) - 1.0f;
        }

        steady_clock::time_point start = steady_clock::now( );

        for ( size_t i = 0; i < ITERATIONS_COUNT; i++ )
        {
            vectorTools->Max( src.data( ), alpha, dst.data( ), src.size( ) );
        }

        auto timeTaken = duration_cast<std::chrono::milliseconds>( steady_clock::now( ) - start ).count( );

        printf( "time taken: %u \n", static_cast<uint32_t>( timeTaken ) );
        for ( size_t i = 0; i < 8; i++ )
        {
            printf( "%f ", static_cast<float>( dst[i] ) );
        }
        printf( "\n" );
        for ( size_t i = 0; i < 8; i++ )
        {
            printf( "%f ", static_cast<float>( dst[VECTOR_SIZE - 8 + i] ) );
        }
        printf( "\n" );

        avgTime += static_cast<float>( timeTaken );
    }

    avgTime /= TESTS_COUNT;

    return avgTime;
}
