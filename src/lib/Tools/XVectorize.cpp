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

#include "XVectorize.hpp"
#include "XAvxVectorTools.hpp"
#include "XSseVectorTools.hpp"
#include "XVectorTools.hpp"
#include "../Config.hpp"

namespace ANNT {

// Get vectorization tools available on current CPU architecture
IVectorTools* GetAvailableVectorTools( )
{
    IVectorTools* vectorTools = nullptr;

#ifdef ANNT_USE_AVX
    if ( vectorTools == nullptr )
    {
        vectorTools = new XAvxVectorTools( );

        if ( !vectorTools->IsAvailable( ) )
        {
            delete vectorTools;
            vectorTools = nullptr;
        }
    }
#endif

#ifdef ANNT_USE_SSE
    if ( vectorTools == nullptr )
    {
        vectorTools = new XSseVectorTools( );

        if ( !vectorTools->IsAvailable( ) )
        {
            delete vectorTools;
            vectorTools = nullptr;
        }
    }
#endif

    if ( vectorTools == nullptr )
    {
        vectorTools = new XVectorTools( );
    }

    return vectorTools;
}

// Initialize vectorizer with what is available
IVectorTools* XVectorize::mVectorTools = GetAvailableVectorTools( );

} // namespace ANNT
