//          Copyright John McFarlane 2015 - 2016.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file ../LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

// definitions that are directly required by more than one header of the API

#if !defined(SG14_COMMON_H)
#define SG14_COMMON_H 1

namespace sg14 {
    namespace _impl {
        ////////////////////////////////////////////////////////////////////////////////
        // sg14::_impl::max

        template<class T>
        constexpr T max(T a, T b)
        {
            return (a<b) ? b : a;
        }

        ////////////////////////////////////////////////////////////////////////////////
        // sg14::comparison_policy

        template<class Lhs, class Rhs, class _Enable = void>
        struct comparison_policy;
    }

    // operator== for types for which sg14::comparison_policy is defined
    template<class Lhs, class Rhs>
    constexpr auto operator==(const Lhs& lhs, const Rhs& rhs)
    -> decltype(_impl::comparison_policy<Lhs, Rhs>::param(lhs)
            ==_impl::comparison_policy<Lhs, Rhs>::param(rhs))
    {
        return _impl::comparison_policy<Lhs, Rhs>::param(lhs)
                ==_impl::comparison_policy<Lhs, Rhs>::param(rhs);
    }

    // operator!= for types for which sg14::comparison_policy is defined
    template<class Lhs, class Rhs>
    constexpr auto operator!=(const Lhs& lhs, const Rhs& rhs)
    -> decltype(_impl::comparison_policy<Lhs, Rhs>::param(lhs)
            !=_impl::comparison_policy<Lhs, Rhs>::param(rhs))
    {
        return _impl::comparison_policy<Lhs, Rhs>::param(lhs)
                !=_impl::comparison_policy<Lhs, Rhs>::param(rhs);
    }

    // operator< for types for which sg14::comparison_policy is defined
    template<class Lhs, class Rhs>
    constexpr auto operator<(const Lhs& lhs, const Rhs& rhs)
    -> decltype(_impl::comparison_policy<Lhs, Rhs>::param(lhs)
            <_impl::comparison_policy<Lhs, Rhs>::param(rhs))
    {
        return _impl::comparison_policy<Lhs, Rhs>::param(lhs)
                <_impl::comparison_policy<Lhs, Rhs>::param(rhs);
    }

    // operator> for types for which sg14::comparison_policy is defined
    template<class Lhs, class Rhs>
    constexpr auto operator>(const Lhs& lhs, const Rhs& rhs)
    -> decltype(_impl::comparison_policy<Lhs, Rhs>::param(lhs)
            >_impl::comparison_policy<Lhs, Rhs>::param(rhs))
    {
        return _impl::comparison_policy<Lhs, Rhs>::param(lhs)
                >_impl::comparison_policy<Lhs, Rhs>::param(rhs);
    }

    // operator<= for types for which sg14::comparison_policy is defined
    template<class Lhs, class Rhs>
    constexpr auto operator<=(const Lhs& lhs, const Rhs& rhs)
    -> decltype(_impl::comparison_policy<Lhs, Rhs>::param(lhs)
            <=_impl::comparison_policy<Lhs, Rhs>::param(rhs))
    {
        return _impl::comparison_policy<Lhs, Rhs>::param(lhs)
                <=_impl::comparison_policy<Lhs, Rhs>::param(rhs);
    }

    // operator>= for types for which sg14::comparison_policy is defined
    template<class Lhs, class Rhs>
    constexpr auto operator>=(const Lhs& lhs, const Rhs& rhs)
    -> decltype(_impl::comparison_policy<Lhs, Rhs>::param(lhs)
            >=_impl::comparison_policy<Lhs, Rhs>::param(rhs))
    {
        return _impl::comparison_policy<Lhs, Rhs>::param(lhs)
                >=_impl::comparison_policy<Lhs, Rhs>::param(rhs);
    }
}

#endif  // SG14_COMMON_H
