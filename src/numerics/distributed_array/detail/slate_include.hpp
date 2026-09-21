#ifndef NUMERICS_DISTRIBUTED_ARRAY_SLATE_INCLUDE_HPP
#define NUMERICS_DISTRIBUTED_ARRAY_SLATE_INCLUDE_HPP

/*
 * SLATE's `roundup` function can conflict with an OS-provided `roundup` macro,
 * so we explicitly undefine the macro here just in case.
 * See https://github.com/icl-utk-edu/slate/issues/227.
 */
#ifdef roundup
#undef roundup
#endif

#include "slate/slate.hh"

#endif // NUMERICS_DISTRIBUTED_ARRAY_SLATE_INCLUDE_HPP
