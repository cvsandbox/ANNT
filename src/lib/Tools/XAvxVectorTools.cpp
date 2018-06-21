/*
    ANNT - Artificial Neural Networks C++ library

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

#include <type_traits>

#ifdef _MSC_VER
    #include <intrin.h>
#elif __GNUC__
    #include <x86intrin.h>
#endif

#include "XAvxVectorTools.hpp"
#include "XCpu.hpp"

#include "../Config.hpp"

namespace ANNT {

// Helper class wrapping some AVX intrinsics
class AvxTools
{
public:
    // Add two vectors: dst[i] += src[i]
    template <typename T> static inline void Add( const T* src, T* dst, size_t size )
    {
        if ( IsAligned( src ) )
        {
            if ( IsAligned( dst ) )
            {
                Add<T, std::true_type, std::true_type>( src, dst, size );
            }
            else
            {
                Add<T, std::true_type, std::false_type>( src, dst, size );
            }
        }
        else
        {
            if ( IsAligned( dst ) )
            {
                Add<T, std::false_type, std::true_type>( src, dst, size );
            }
            else
            {
                Add<T, std::false_type, std::false_type>( src, dst, size );
            }
        }
    }

    // Multiply two vectors: dst[i] *= src[i]
    template <typename T> static inline void Mul( const T* src, T* dst, size_t size )
    {
        if ( IsAligned( src ) )
        {
            if ( IsAligned( dst ) )
            {
                Mul<T, std::true_type, std::true_type>( src, dst, size );
            }
            else
            {
                Mul<T, std::true_type, std::false_type>( src, dst, size );
            }
        }
        else
        {
            if ( IsAligned( dst ) )
            {
                Mul<T, std::false_type, std::true_type>( src, dst, size );
            }
            else
            {
                Mul<T, std::false_type, std::false_type>( src, dst, size );
            }
        }
    }

    // Dot product: sum( vec1[i] * vec2[i] )
    template <typename T> static inline T Dot( const T* vec1, const T* vec2, size_t size )
    {
        T dotProduct;

        if ( IsAligned( vec1 ) )
        {
            if ( IsAligned( vec2 ) )
            {
                dotProduct = Dot<T, std::true_type, std::true_type>( vec1, vec2, size );
            }
            else
            {
                dotProduct = Dot<T, std::true_type, std::false_type>( vec1, vec2, size );
            }
        }
        else
        {
            if ( IsAligned( vec2 ) )
            {
                dotProduct = Dot<T, std::false_type, std::true_type>( vec1, vec2, size );
            }
            else
            {
                dotProduct = Dot<T, std::false_type, std::false_type>( vec1, vec2, size );
            }
        }

        return dotProduct;
    }

    // Calculates maximum of the vector's elements and the specified value: dst[i] = max( src[i], alpha )
    template <typename T> static inline void Max( const T* src, T alpha, T* dst, size_t size )
    {
        if ( IsAligned( src ) )
        {
            if ( IsAligned( dst ) )
            {
                Max<T, std::true_type, std::true_type>( src, alpha, dst, size );
            }
            else
            {
                Max<T, std::true_type, std::false_type>( src, alpha, dst, size );
            }
        }
        else
        {
            if ( IsAligned( dst ) )
            {
                Max<T, std::false_type, std::true_type>( src, alpha, dst, size );
            }
            else
            {
                Max<T, std::false_type, std::false_type>( src, alpha, dst, size );
            }
        }
    }

private:
    // Unroll size for single/double precision numbers - number of those in AVX register
    template <typename T> static inline size_t UnrollSize( );

    // Load 8 single / 4 double precision numbers into AVX register
    template <typename isAligned> static inline __m256  Load( const float*  src );
    template <typename isAligned> static inline __m256d Load( const double* src );

    // Store 8 single / 4 double precision numbers from AVX register into memory
    template <typename isAligned> static inline void Store( const __m256&  value, float*  dst );
    template <typename isAligned> static inline void Store( const __m256d& value, double* dst );

    // Check if the pointer is AVX-aligned (32 byte aligned)
    template <typename T> static inline bool IsAligned( const T* ptr )
    {
        return ( ( reinterpret_cast<uintptr_t>( ptr ) % 32 ) == 0 );
    }

    // Initialize 8 single / 4 double precision numbers of AVX register with the specified value 
    static inline __m256 Set1( float value )
    {
        return _mm256_set1_ps( value );
    }
    static inline __m256d Set1( double value )
    {
        return _mm256_set1_pd( value );
    }

    // Sum 8 single / 4 double precision numbers of AVX register
    static inline float  Sum( __m256 value );
    static inline double Sum( __m256d value );

    // Add 8 single / 4 double precision numbers
    static inline __m256 Add( const __m256& value1, const __m256& value2 )
    {
        return _mm256_add_ps( value1, value2 );
    }
    static inline __m256d Add( const __m256d& value1, const __m256d& value2 )
    {
        return _mm256_add_pd( value1, value2 );
    }

    // Multiple 8 single / 4 double precision numbers
    static inline __m256 Mul( const __m256& value1, const __m256& value2 )
    {
        return _mm256_mul_ps( value1, value2 );
    }
    static inline __m256d Mul( const __m256d& value1, const __m256d& value2 )
    {
        return _mm256_mul_pd( value1, value2 );
    }

    // Multiple and Add 8 single / 4 double precision numbers: value1 * value2 + value3
    static inline __m256 MAdd( const __m256& value1, const __m256& value2, const __m256& value3 )
    {
#ifdef USE_AVX2
        return _mm256_fmadd_ps( value1, value2, value3 );
#else
        return _mm256_add_ps( _mm256_mul_ps( value1, value2 ), value3 );
#endif
    }
    static inline __m256d MAdd( const __m256d& value1, const __m256d& value2, const __m256d& value3 )
    {
#ifdef USE_AVX2
        return _mm256_fmadd_pd( value1, value2, value3 );
#else
        return _mm256_add_pd( _mm256_mul_pd( value1, value2 ), value3 );
#endif
    }

    // Maximum of 8 single / 4 double precision numbers
    static inline __m256 Max( const __m256& value1, const __m256& value2 )
    {
        return _mm256_max_ps( value1, value2 );
    }
    static inline __m256d Max( const __m256d& value1, const __m256d& value2 )
    {
        return _mm256_max_pd( value1, value2 );
    }

    // Add two vectors
    template <typename T, typename srcAligned, typename dstAligned> static void Add( const T* src, T* dst, size_t size )
    {
        size_t blockSize        = UnrollSize<T>( );
        size_t blockSize2       = blockSize * 2;
        size_t blockSize3       = blockSize * 3;
        size_t blockSize4       = blockSize * 4;
        size_t blockIterations4 = size / blockSize4;
        size_t blockIterations  = ( size - blockIterations4 * blockSize4 ) / blockSize;
        size_t remainIterations = size - blockIterations4 * blockSize4 - blockIterations * blockSize;

        // large blocks of 4
        for ( size_t i = 0; i < blockIterations4; i++ )
        {
            auto s0 = Load<srcAligned>(  src );
            auto s1 = Load<srcAligned>( &src[blockSize ] );
            auto s2 = Load<srcAligned>( &src[blockSize2] );
            auto s3 = Load<srcAligned>( &src[blockSize3] );
            auto d0 = Load<dstAligned>(  dst );
            auto d1 = Load<dstAligned>( &dst[blockSize ] );
            auto d2 = Load<dstAligned>( &dst[blockSize2] );
            auto d3 = Load<dstAligned>( &dst[blockSize3] );

            d0 = Add( s0, d0 );
            d1 = Add( s1, d1 );
            d2 = Add( s2, d2 );
            d3 = Add( s3, d3 );

            Store<dstAligned>( d0,  dst );
            Store<dstAligned>( d1, &dst[blockSize ] );
            Store<dstAligned>( d2, &dst[blockSize2] );
            Store<dstAligned>( d3, &dst[blockSize3] );

            src += blockSize4;
            dst += blockSize4;
        }

        // small blocks of 1
        for ( size_t i = 0; i < blockIterations; i++ )
        {
            auto s = Load<srcAligned>( src );
            auto d = Load<dstAligned>( dst );

            d = Add( s, d );

            Store<dstAligned>( d, dst );

            src += blockSize;
            dst += blockSize;
        }

        // remainder for compiler to decide
        for ( size_t i = 0; i < remainIterations; i++ )
        {
            *dst += *src;

            src++;
            dst++;
        }
    }

    // Multiply two vectors
    template <typename T, typename srcAligned, typename dstAligned> static void Mul( const T* src, T* dst, size_t size )
    {
        size_t blockSize        = UnrollSize<T>( );
        size_t blockSize2       = blockSize * 2;
        size_t blockSize3       = blockSize * 3;
        size_t blockSize4       = blockSize * 4;
        size_t blockIterations4 = size / blockSize4;
        size_t blockIterations  = ( size - blockIterations4 * blockSize4 ) / blockSize;
        size_t remainIterations = size - blockIterations4 * blockSize4 - blockIterations * blockSize;

        // large blocks of 4
        for ( size_t i = 0; i < blockIterations4; i++ )
        {
            auto s0 = Load<srcAligned>(  src );
            auto s1 = Load<srcAligned>( &src[blockSize ] );
            auto s2 = Load<srcAligned>( &src[blockSize2] );
            auto s3 = Load<srcAligned>( &src[blockSize3] );
            auto d0 = Load<dstAligned>(  dst );
            auto d1 = Load<dstAligned>( &dst[blockSize ] );
            auto d2 = Load<dstAligned>( &dst[blockSize2] );
            auto d3 = Load<dstAligned>( &dst[blockSize3] );

            d0 = Mul( s0, d0 );
            d1 = Mul( s1, d1 );
            d2 = Mul( s2, d2 );
            d3 = Mul( s3, d3 );

            Store<dstAligned>( d0,  dst );
            Store<dstAligned>( d1, &dst[blockSize ] );
            Store<dstAligned>( d2, &dst[blockSize2] );
            Store<dstAligned>( d3, &dst[blockSize3] );

            src += blockSize4;
            dst += blockSize4;
        }

        // small blocks of 1
        for ( size_t i = 0; i < blockIterations; i++ )
        {
            auto s = Load<srcAligned>( src );
            auto d = Load<dstAligned>( dst );

            d = Mul( s, d );

            Store<dstAligned>( d, dst );

            src += blockSize;
            dst += blockSize;
        }

        // remainder for compiler to decide
        for ( size_t i = 0; i < remainIterations; i++ )
        {
            *dst *= *src;

            src++;
            dst++;
        }
    }

    // Dot product of two vectors
    template <typename T, typename vec1Aligned, typename vec2Aligned> static T Dot( const T* vec1, const T* vec2, size_t size )
    {
        size_t blockSize        = UnrollSize<T>( );
        size_t blockSize2       = blockSize * 2;
        size_t blockSize3       = blockSize * 3;
        size_t blockSize4       = blockSize * 4;
        size_t blockIterations4 = size / blockSize4;
        size_t blockIterations  = ( size - blockIterations4 * blockSize4 ) / blockSize;
        size_t remainIterations = size - blockIterations4 * blockSize4 - blockIterations * blockSize;
        
        auto   sum0 = Set1( T( 0 ) );
        auto   sum1 = Set1( T( 0 ) );
        auto   sum2 = Set1( T( 0 ) );
        auto   sum3 = Set1( T( 0 ) );

        // large blocks of 4
        for ( size_t i = 0; i < blockIterations4; i++ )
        {
            auto v10 = Load<vec1Aligned>(  vec1 );
            auto v11 = Load<vec1Aligned>( &vec1[blockSize ] );
            auto v12 = Load<vec1Aligned>( &vec1[blockSize2] );
            auto v13 = Load<vec1Aligned>( &vec1[blockSize3] );
            auto v20 = Load<vec2Aligned>(  vec2 );
            auto v21 = Load<vec2Aligned>( &vec2[blockSize ] );
            auto v22 = Load<vec2Aligned>( &vec2[blockSize2] );
            auto v23 = Load<vec2Aligned>( &vec2[blockSize3] );

            sum0 = MAdd( v10, v20, sum0 );
            sum1 = MAdd( v11, v21, sum1 );
            sum2 = MAdd( v12, v22, sum2 );
            sum3 = MAdd( v13, v23, sum3 );

            vec1 += blockSize4;
            vec2 += blockSize4;
        }

        // small blocks of 1
        for ( size_t i = 0; i < blockIterations; i++ )
        {
            auto v1 = Load<vec1Aligned>( vec1 );
            auto v2 = Load<vec2Aligned>( vec2 );

            sum0 = MAdd( v1, v2, sum0 );

            vec1 += blockSize;
            vec2 += blockSize;
        }

        sum0  = Add( sum0, sum1 );
        sum0  = Add( sum0, sum2 );
        sum0  = Add( sum0, sum3 );

        T sum = Sum( sum0 );

        for ( size_t i = 0; i < remainIterations; i++ )
        {
            sum += *vec1 * *vec2;

            vec1++;
            vec2++;
        }

        return sum;
    }

    // Maximum value of vector's elements and the specified alpha value
    template <typename T, typename srcAligned, typename dstAligned> static void Max( const T* src, T alpha, T* dst, size_t size )
    {
        size_t blockSize        = UnrollSize<T>( );
        size_t blockSize2       = blockSize * 2;
        size_t blockSize3       = blockSize * 3;
        size_t blockSize4       = blockSize * 4;
        size_t blockIterations4 = size / blockSize4;
        size_t blockIterations  = ( size - blockIterations4 * blockSize4 ) / blockSize;
        size_t remainIterations = size - blockIterations4 * blockSize4 - blockIterations * blockSize;

        auto   alphaVec = Set1( alpha );

        // large blocks of 4
        for ( size_t i = 0; i < blockIterations4; i++ )
        {
            auto s0 = Load<srcAligned>(  src );
            auto s1 = Load<srcAligned>( &src[blockSize ] );
            auto s2 = Load<srcAligned>( &src[blockSize2] );
            auto s3 = Load<srcAligned>( &src[blockSize3] );

            s0 = Max( s0, alphaVec );
            s1 = Max( s1, alphaVec );
            s2 = Max( s2, alphaVec );
            s3 = Max( s3, alphaVec );

            Store<dstAligned>( s0,  dst );
            Store<dstAligned>( s1, &dst[blockSize ] );
            Store<dstAligned>( s2, &dst[blockSize2] );
            Store<dstAligned>( s3, &dst[blockSize3] );

            src += blockSize4;
            dst += blockSize4;
        }

        // small blocks of 1
        for ( size_t i = 0; i < blockIterations; i++ )
        {
            auto s = Load<srcAligned>( src );

            s = Max( s, alphaVec );

            Store<dstAligned>( s, dst );

            src += blockSize;
            dst += blockSize;
        }

        // remainder for compiler to decide
        for ( size_t i = 0; i < remainIterations; i++ )
        {
            *dst = ( *src > alpha ) ? *src : alpha;

            src++;
            dst++;
        }
    }
};

// Unroll size for single/double precision numbers - number of those in AVX register
template <> inline size_t AvxTools::UnrollSize<float>( )
{
    return 8;
}
template <> inline size_t AvxTools::UnrollSize<double>( )
{
    return 4;
}

// Load 8 aligned single precision numbers
template <> inline __m256 AvxTools::Load<std::true_type>( const float* src )
{
    return _mm256_load_ps( src );
}

// Load 8 unaligned single precision numbers
template <> inline __m256 AvxTools::Load<std::false_type>( const float* src )
{
    return _mm256_loadu_ps( src );
}

// Load 4 aligned double precision numbers
template <> inline __m256d AvxTools::Load<std::true_type>( const double* src )
{
    return _mm256_load_pd( src );
}

// Load 4 unaligned double precision numbers
template <> inline __m256d AvxTools::Load<std::false_type>( const double* src )
{
    return _mm256_loadu_pd( src );
}

// Store 8 signle precision numbers into aligned memory
template <> inline void AvxTools::Store<std::true_type>( const __m256& value, float* dst )
{
    _mm256_store_ps( dst, value );
}

// Store 8 signle precision numbers into unaligned memory
template <> inline void AvxTools::Store<std::false_type>( const __m256& value, float* dst )
{
    _mm256_storeu_ps( dst, value );
}

// Store 4 double precision numbers into aligned memory
template <> inline void AvxTools::Store<std::true_type>( const __m256d& value, double* dst )
{
    _mm256_store_pd( dst, value );
}

// Store 4 double precision numbers into unaligned memory
template <> inline void AvxTools::Store<std::false_type>( const __m256d& value, double* dst )
{
    _mm256_storeu_pd( dst, value );
}

// Sum 8 single / 4 double precision numbers of AVX register
inline float AvxTools::Sum( __m256 value )
{
    float mem[8];

    Store<std::false_type>( value, mem );

    return mem[0] + mem[1] + mem[2] + mem[3] + mem[4] + mem[5] + mem[6] + mem[7];
}
inline double AvxTools::Sum( __m256d value )
{
    double mem[4];

    Store<std::false_type>( value, mem );

    return mem[0] + mem[1] + mem[2] + mem[3];
}

/* ============================================================================= */

// Check if the implementation of vector tools is available on the current system
bool XAvxVectorTools::IsAvailable( ) const
{
#if defined(ANNT_USE_AVX2)
    return XCpu::IsFeatureSupported( XCpu::Reg_EBX, XCpu::Flag_AVX2 );
#elif defined(ANNT_USE_AVX)
    return XCpu::IsFeatureSupported( XCpu::Reg_ECX, XCpu::Flag_AVX );
#else
    return false;
#endif
}

// Add two vectors: dst[i] += src[i]
void XAvxVectorTools::Add( const float* src, float* dst, size_t size ) const
{
    AvxTools::Add( src, dst, size );
}
void XAvxVectorTools::Add( const double* src, double* dst, size_t size ) const
{
    AvxTools::Add( src, dst, size );
};

// Element wise multiplication of two vectors (Hadamard product): dst[i] *= src[i]
void XAvxVectorTools::Mul( const float*  src, float*  dst, size_t size ) const
{
    AvxTools::Mul( src, dst, size );
}
void XAvxVectorTools::Mul( const double* src, double* dst, size_t size ) const
{
    AvxTools::Mul( src, dst, size );
}

// Dot product of two vectors: sum( vec1[i] * vec2[i] )
float XAvxVectorTools::Dot( const float* vec1, const float* vec2, size_t size ) const
{
    return AvxTools::Dot( vec1, vec2, size );
}
double XAvxVectorTools::Dot( const double* vec1, const double* vec2, size_t size ) const
{
    return AvxTools::Dot( vec1, vec2, size );
}

// Calculates maximum of the vector's elements and the specified value: dst[i] = max( src[i], alpha )
void XAvxVectorTools::Max( const float* src, float alpha, float* dst, size_t size ) const
{
    AvxTools::Max( src, alpha, dst, size );
}
void XAvxVectorTools::Max( const double* src, double alpha, double* dst, size_t size ) const
{
    AvxTools::Max( src, alpha, dst, size );
}

} // namespace ANNT
