
//          Copyright John McFarlane 2015 - 2016.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file ../LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// \brief essential definitions related to the `sg14::fixed_point` type

#if !defined(SG14_FIXED_POINT_H)
#define SG14_FIXED_POINT_H 1

#include "type_traits.h"

#include "bits/common.h"

////////////////////////////////////////////////////////////////////////////////
// SG14_FIXED_POINT_EXCEPTIONS_ENABLED macro definition

#if defined(SG14_FIXED_POINT_EXCEPTIONS_ENABLED)
#error SG14_FIXED_POINT_EXCEPTIONS_ENABLED already defined
#endif

#if defined(_MSC_VER)
#if defined(_CPPUNWIND)
#define SG14_FIXED_POINT_EXCEPTIONS_ENABLED
#endif
#elif defined(__clang__) || defined(__GNUG__)
#if defined(__EXCEPTIONS)
#define SG14_FIXED_POINT_EXCEPTIONS_ENABLED
#endif
#else
#define SG14_FIXED_POINT_EXCEPTIONS_ENABLED
#endif

#if defined(SG14_FIXED_POINT_EXCEPTIONS_ENABLED)

#include <stdexcept>

#endif

/// study group 14 of the C++ working group
namespace sg14 {
    ////////////////////////////////////////////////////////////////////////////////
    // general-purpose _fixed_point_impl definitions

    namespace _fixed_point_impl {
        ////////////////////////////////////////////////////////////////////////////////
        // sg14::_fixed_point_impl::digits

        template<class T>
        using digits = std::integral_constant<int, width<T>::value-is_signed<T>::value>;

        ////////////////////////////////////////////////////////////////////////////////
        // sg14::_fixed_point_impl::float_of_same_size

        template<class T>
        using float_of_same_size = set_width_t<float, width<T>::value>;

        ////////////////////////////////////////////////////////////////////////////////
        // sg14::_fixed_point_impl::next_size

        // given an integral type, IntType,
        // provides the integral type of the equivalent type with twice the width
        template<class IntType>
        using next_size = typename sg14::set_width_t<IntType, width<IntType>::value*2>;

        ////////////////////////////////////////////////////////////////////////////////
        // sg14::_fixed_point_impl::previous_size

        // given an integral type, IntType,
        // provides the integral type of the equivalent type with half the width
        template<class IntType>
        using previous_size = typename sg14::set_width_t<IntType, width<IntType>::value/2>;

        ////////////////////////////////////////////////////////////////////////////////
        // sg14::_fixed_point_impl::pow2

        // returns given power of 2
        template<class S, int Exponent, typename std::enable_if<Exponent==0, int>::type Dummy = 0>
        constexpr S pow2()
        {
            static_assert(std::is_floating_point<S>::value, "S must be floating-point type");
            return 1;
        }

        template<class S, int Exponent, typename std::enable_if<!(Exponent<=0) && (Exponent<8), int>::type Dummy = 0>
        constexpr S pow2()
        {
            static_assert(std::is_floating_point<S>::value, "S must be floating-point type");
            return pow2<S, Exponent-1>()*S(2);
        }

        template<class S, int Exponent, typename std::enable_if<(Exponent>=8), int>::type Dummy = 0>
        constexpr S pow2()
        {
            static_assert(std::is_floating_point<S>::value, "S must be floating-point type");
            return pow2<S, Exponent-8>()*S(256);
        }

        template<class S, int Exponent, typename std::enable_if<!(Exponent>=0) && (Exponent>-8), int>::type Dummy = 0>
        constexpr S pow2()
        {
            static_assert(std::is_floating_point<S>::value, "S must be floating-point type");
            return pow2<S, Exponent+1>()*S(.5);
        }

        template<class S, int Exponent, typename std::enable_if<(Exponent<=-8), int>::type Dummy = 0>
        constexpr S pow2()
        {
            static_assert(std::is_floating_point<S>::value, "S must be floating-point type");
            return pow2<S, Exponent+8>()*S(.003906250);
        }

        ////////////////////////////////////////////////////////////////////////////////
        // sg14::_fixed_point_impl::shift_left and sg14::_fixed_point_impl::shift_right

        // performs a shift operation by a fixed number of bits avoiding two pitfalls:
        // 1) shifting by a negative amount causes undefined behavior
        // 2) converting between integer types of different sizes can lose significant bits during shift right

        // Exponent == 0
        template<
                int Exponent,
                class Output,
                class Input,
                typename std::enable_if<
                        (Exponent==0),
                        int>::type Dummy = 0>
        constexpr Output shift_left(Input i)
        {
            // cast only
            return static_cast<Output>(i);
        }

        template<
                int Exponent,
                class Output,
                class Input,
                typename std::enable_if<
                        Exponent==0,
                        int>::type Dummy = 0>
        constexpr Output shift_right(Input i)
        {
            // cast only
            return static_cast<Output>(i);
        }

        // Exponent >= 0
        template<
                int Exponent,
                class Output,
                class Input,
                typename std::enable_if<
                        !(Exponent<=0),
                        int>::type Dummy = 0>
        constexpr Output shift_left(Input i)
        {
            using larger = typename std::conditional<
                    width<Input>::value<=width<Output>::value,
                    Output, Input>::type;
            return static_cast<Output>(static_cast<larger>(i)*(larger{1} << Exponent));
        }

        template<
                int Exponent,
                class Output,
                class Input,
                typename std::enable_if<
                        !(Exponent<=0),
                        int>::type Dummy = 0>
        constexpr Output shift_right(Input i)
        {
            using larger = typename std::conditional<
                    width<Input>::value<=width<Output>::value,
                    Output, Input>::type;
            return static_cast<Output>(static_cast<larger>(i)/(larger{1} << Exponent));
        }

        // Exponent < 0
        template<
                int Exponent,
                class Output,
                class Input,
                typename std::enable_if<
                        (Exponent<0),
                        int>::type Dummy = 0>
        constexpr Output shift_left(Input i)
        {
            // negate Exponent and flip from left to right
            return shift_right<-Exponent, Output, Input>(i);
        }

        template<
                int Exponent,
                class Output,
                class Input,
                typename std::enable_if<
                        Exponent<0,
                        int>::type Dummy = 0>
        constexpr Output shift_right(Input i)
        {
            // negate Exponent and flip from right to left
            return shift_left<-Exponent, Output, Input>(i);
        }

        ////////////////////////////////////////////////////////////////////////////////
        // sg14::_fixed_point_impl::capacity

        // has value that, given a value N,
        // returns number of bits necessary to represent it in binary
        template<unsigned N>
        struct capacity;

        template<>
        struct capacity<0> {
            static constexpr int value = 0;
        };

        template<unsigned N>
        struct capacity {
            static constexpr int value = capacity<N/2>::value+1;
        };
    }

    /// \brief literal real number approximation that uses fixed-point arithmetic
    ///
    /// \tparam Rep the underlying type used to represent the value
    /// \tparam Exponent the value by which to scale the integer value in order to get the real value
    ///
    /// \par Examples
    ///
    /// To define a fixed-point value 1 byte in size with a sign bit, 3 integer bits and 4 fractional bits:
    /// \snippet snippets.cpp define a fixed_point value

    template<class Rep = int, int Exponent = 0>
    class fixed_point {
    public:
        ////////////////////////////////////////////////////////////////////////////////
        // types

        /// alias to template parameter, \a Rep
        using rep = Rep;

        ////////////////////////////////////////////////////////////////////////////////
        // constants

        /// value of template parameter, \a Exponent
        constexpr static int exponent = Exponent;

        /// number of binary digits this type can represent;
        /// equivalent to [std::numeric_limits::digits](http://en.cppreference.com/w/cpp/types/numeric_limits/digits)
        constexpr static int digits = _fixed_point_impl::digits<Rep>::value;

        /// number of binary digits devoted to integer part of value;
        /// can be negative for specializations with especially small ranges
        constexpr static int integer_digits = digits+exponent;

        /// number of binary digits devoted to fractional part of value;
        /// can be negative for specializations with especially large ranges
        constexpr static int fractional_digits = -exponent;

        ////////////////////////////////////////////////////////////////////////////////
        // functions

    private:
        // constructor taking representation explicitly using operator++(int)-style trick
        constexpr fixed_point(rep r, int)
                :_r(r)
        {
        }

    public:
        /// default constructor
        fixed_point() { }

        /// constructor taking an integer type
        template<class S, typename std::enable_if<is_integral<S>::value, int>::type Dummy = 0>
        explicit constexpr fixed_point(S s)
                :_r(integral_to_rep(s))
        {
        }

        /// constructor taking a floating-point type
        template<class S, typename std::enable_if<std::is_floating_point<S>::value, int>::type Dummy = 0>
        explicit constexpr fixed_point(S s)
                :_r(floating_point_to_rep(s))
        {
        }

        /// constructor taking a fixed-point type
        template<class FromRep, int FromExponent>
        explicit constexpr fixed_point(const fixed_point<FromRep, FromExponent>& rhs)
                :_r(fixed_point_to_rep(rhs))
        {
        }

        /// copy assignment operator taking an integer type
        template<class S, typename std::enable_if<is_integral<S>::value, int>::type Dummy = 0>
        fixed_point& operator=(S s)
        {
            _r = integral_to_rep(s);
            return *this;
        }

        /// copy assignment operator taking a floating-point type
        template<class S, typename std::enable_if<std::is_floating_point<S>::value, int>::type Dummy = 0>
        fixed_point& operator=(S s)
        {
            _r = floating_point_to_rep(s);
            return *this;
        }

        /// copy assignement operator taking a fixed-point type
        template<class FromRep, int FromExponent>
        fixed_point& operator=(const fixed_point<FromRep, FromExponent>& rhs)
        {
            _r = fixed_point_to_rep(rhs);
            return *this;
        }

        /// returns value represented as integral
        template<class S, typename std::enable_if<is_integral<S>::value, int>::type Dummy = 0>
        explicit constexpr operator S() const
        {
            return rep_to_integral<S>(_r);
        }

        /// returns value represented as floating-point
        template<class S, typename std::enable_if<std::is_floating_point<S>::value, int>::type Dummy = 0>
        explicit constexpr operator S() const
        {
            return rep_to_floating_point<S>(_r);
        }

        /// returns non-zeroness represented as boolean
        explicit constexpr operator bool() const
        {
            return _r!=0;
        }

        template<class Rhs>
        fixed_point& operator*=(const Rhs& rhs);

        template<class Rhs>
        fixed_point& operator/=(const Rhs& rhs);

        /// returns internal representation of value
        constexpr rep data() const
        {
            return _r;
        }

        /// creates an instance given the underlying representation value
        static constexpr fixed_point from_data(rep r)
        {
            return fixed_point(r, 0);
        }

    private:
        template<class S, typename std::enable_if<std::is_floating_point<S>::value, int>::type Dummy = 0>
        static constexpr S one()
        {
            return _fixed_point_impl::pow2<S, -exponent>();
        }

        template<class S, typename std::enable_if<is_integral<S>::value, int>::type Dummy = 0>
        static constexpr S one()
        {
            return integral_to_rep<S>(1);
        }

        template<class S>
        static constexpr S inverse_one()
        {
            static_assert(std::is_floating_point<S>::value, "S must be floating-point type");
            return _fixed_point_impl::pow2<S, exponent>();
        }

        template<class S>
        static constexpr rep integral_to_rep(S s)
        {
            static_assert(is_integral<S>::value, "S must be unsigned integral type");

            return _fixed_point_impl::shift_right<exponent, rep>(s);
        }

        template<class S>
        static constexpr S rep_to_integral(rep r)
        {
            static_assert(is_integral<S>::value, "S must be unsigned integral type");

            return _fixed_point_impl::shift_left<exponent, S>(r);
        }

        template<class S>
        static constexpr rep floating_point_to_rep(S s)
        {
            static_assert(std::is_floating_point<S>::value, "S must be floating-point type");
            return static_cast<rep>(s*one<S>());
        }

        template<class S>
        static constexpr S rep_to_floating_point(rep r)
        {
            static_assert(std::is_floating_point<S>::value, "S must be floating-point type");
            return S(r)*inverse_one<S>();
        }

        template<class FromRep, int FromExponent>
        static constexpr rep fixed_point_to_rep(const fixed_point<FromRep, FromExponent>& rhs)
        {
            return _fixed_point_impl::shift_right<(exponent-FromExponent), rep>(rhs.data());
        }

        ////////////////////////////////////////////////////////////////////////////////
        // variables

        rep _r;
    };

    /// \brief Produce a fixed-point type with the given number of integer and fractional digits.
    ///
    /// \tparam IntegerDigits specifies minimum value of @ref fixed_point::integer_digits
    /// \tparam FractionalDigits specifies the exact value of @ref fixed_point::fractional_digits
    /// \tparam Archetype hints at the type of @ref fixed_point::rep
    ///
    /// \remarks The signage of \a Archetype specifies signage of the resultant fixed-point type.
    /// \remarks Typical choices for \a Archetype, `signed` and `unsigned`,
    /// result in a type that uses built-in integers for \a fixed_point::rep.
    /// \remarks Resultant type is signed by default.
    ///
    /// \par Example:
    ///
    /// To generate a fixed-point type with a sign bit, 8 fractional bits and at least 7 integer bits:
    /// \snippet snippets.cpp use make_fixed
    ///
    /// \sa make_ufixed
    template<int IntegerDigits, int FractionalDigits = 0, class Archetype = signed>
    using make_fixed = fixed_point<
            set_width_t<Archetype, IntegerDigits+FractionalDigits+is_signed<Archetype>::value>,
            -FractionalDigits>;

    /// \brief Produce an unsigned fixed-point type with the given number of integer and fractional digits.
    ///
    /// \sa make_fixed
    template<int IntegerDigits, int FractionalDigits = 0, class Archetype = unsigned>
    using make_ufixed = make_fixed<
            IntegerDigits,
            FractionalDigits,
            typename make_unsigned<Archetype>::type>;

    ////////////////////////////////////////////////////////////////////////////////
    // sg14::fixed_point-aware _fixed_point_impl definitions

    namespace _fixed_point_impl {
        ////////////////////////////////////////////////////////////////////////////////
        // sg14::_fixed_point_impl::is_fixed_point

        template<class T>
        struct is_fixed_point;

        template<class T>
        struct is_fixed_point
                : public std::false_type {
        };

        template<class Rep, int Exponent>
        struct is_fixed_point<fixed_point<Rep, Exponent>>
                : public std::true_type {
        };

        ////////////////////////////////////////////////////////////////////////////////
        // sg14::_fixed_point_impl::common_type_mixed

        template<class Lhs, class Rhs, class _Enable = void>
        struct common_type_mixed;

        // given a fixed-point and an integer type,
        // generates a fixed-point type that is as big as both of them (or as close as possible)
        template<class LhsRep, int LhsExponent, class RhsInteger>
        struct common_type_mixed<
                fixed_point<LhsRep, LhsExponent>,
                RhsInteger,
                typename std::enable_if<is_integral<RhsInteger>::value>::type>
                : std::common_type<fixed_point<LhsRep, LhsExponent>, fixed_point<RhsInteger, 0>> {
        };

        // given a fixed-point and a floating-point type,
        // generates a floating-point type that is as big as both of them (or as close as possible)
        template<class LhsRep, int LhsExponent, class Float>
        struct common_type_mixed<
                fixed_point<LhsRep, LhsExponent>,
                Float,
                typename std::enable_if<std::is_floating_point<Float>::value>::type>
                : std::common_type<float_of_same_size<LhsRep>, Float> {
        };
    }
}

namespace std {
    ////////////////////////////////////////////////////////////////////////////////
    // std::common_type<> specializations related to sg14::sg14::fixed_point<>

    // std::common_type<fixed_point<>>
    template<class Rep, int Exponent>
    struct common_type<sg14::fixed_point<Rep, Exponent>> {
        using type = sg14::fixed_point<
                typename std::common_type<Rep>::type,
                Exponent>;
    };

    // std::common_type<fixed_point<>, not-fixed-point>
    template<class LhsRep, int LhsExponent, class Rhs>
    struct common_type<sg14::fixed_point<LhsRep, LhsExponent>, Rhs> {
        static_assert(!sg14::_fixed_point_impl::is_fixed_point<Rhs>::value, "fixed-point Rhs type");
        using type = typename sg14::_fixed_point_impl::common_type_mixed<sg14::fixed_point<LhsRep, LhsExponent>, Rhs>::type;
    };

    // std::common_type<not-fixed-point, fixed_point<>>
    template<class Lhs, class RhsRep, int RhsExponent>
    struct common_type<Lhs, sg14::fixed_point<RhsRep, RhsExponent>> {
        static_assert(!sg14::_fixed_point_impl::is_fixed_point<Lhs>::value, "fixed-point Lhs type");
        using type = typename sg14::_fixed_point_impl::common_type_mixed<sg14::fixed_point<RhsRep, RhsExponent>, Lhs>::type;
    };

    // std::common_type<fixed_point<>, fixed_point<>>
    template<class LhsRep, int LhsExponent, class RhsRep, int RhsExponent>
    struct common_type<sg14::fixed_point<LhsRep, LhsExponent>, sg14::fixed_point<RhsRep, RhsExponent>> {
        using _result_rep = typename std::common_type<LhsRep, RhsRep>::type;

        // exponent is the lower of the two operands' unless that could cause overflow in which case it is adjusted downward
        static constexpr int _capacity = sg14::_fixed_point_impl::digits<_result_rep>::value;
        static constexpr int _ideal_max_top = sg14::_impl::max(
                sg14::fixed_point<LhsRep, LhsExponent>::integer_digits,
                sg14::fixed_point<RhsRep, RhsExponent>::integer_digits);
        static constexpr int _ideal_exponent = sg14::_impl::min(LhsExponent, RhsExponent);
        static constexpr int _exponent = ((_ideal_max_top-_ideal_exponent)<=_capacity) ? _ideal_exponent :
                                         _ideal_max_top-_capacity;

        using type = sg14::fixed_point<_result_rep, _exponent>;
    };
}

namespace sg14 {
    ////////////////////////////////////////////////////////////////////////////////
    // named fixed-point arithmetic traits

    namespace _fixed_point_impl {
        template<typename Lhs, typename Rhs>
        struct binary_pair;

        template<class LhsRep, int LhsExponent, class RhsRep, int RhsExponent>
        struct binary_pair<fixed_point<LhsRep, LhsExponent>, fixed_point<RhsRep, RhsExponent>> {
            using lhs_type = fixed_point<LhsRep, LhsExponent>;
            using rhs_type = fixed_point<RhsRep, RhsExponent>;
        };

        template<class LhsRep, int LhsExponent, class Rhs>
        struct binary_pair<fixed_point<LhsRep, LhsExponent>, Rhs> {
            using lhs_type = fixed_point<LhsRep, LhsExponent>;
            using rhs_type = Rhs;
        };

        template<class Lhs, class RhsRep, int RhsExponent>
        struct binary_pair<Lhs, fixed_point<RhsRep, RhsExponent>> {
            using lhs_type = Lhs;
            using rhs_type = fixed_point<RhsRep, RhsExponent>;
        };

        ////////////////////////////////////////////////////////////////////////////////
        // lean arithmetic - as close to machine arithmetic as possible

        template<typename Lhs, typename Rhs, class RepType>
        struct add_subtract_lean {
            using rep_type = RepType;

            static constexpr int result_exponent = (Lhs::integer_digits>Rhs::integer_digits)
                                                   ? Lhs::exponent : Rhs::exponent;

            using result_type = fixed_point<rep_type, result_exponent>;
        };

        template<typename Lhs, typename Rhs>
        struct add_lean : add_subtract_lean<Lhs, Rhs, decltype(std::declval<typename Lhs::rep>()
                +std::declval<typename Rhs::rep>())> {
        };

        template<typename Lhs, typename Rhs>
        struct subtract_lean : add_subtract_lean<Lhs, Rhs, decltype(std::declval<typename Lhs::rep>()
                -std::declval<typename Rhs::rep>())> {
        };

        template<typename Lhs, typename Rhs>
        struct multiply_lean {
            using rep_type = decltype(std::declval<typename Lhs::rep>()*std::declval<typename Rhs::rep>());
            using result_type = fixed_point<rep_type, Lhs::exponent+Rhs::exponent>;
        };

        template<typename Lhs, typename Rhs>
        struct divide_lean {
            using rep_type = decltype(std::declval<typename Lhs::rep>()/std::declval<typename Rhs::rep>());

            static constexpr int wide_result_digits = Lhs::digits+Rhs::digits;
            static constexpr int intermediate_exponent = Lhs::integer_digits-wide_result_digits;

            using intermediate_type = fixed_point<rep_type, intermediate_exponent>;
            using result_type = fixed_point<rep_type, intermediate_type::exponent-Rhs::exponent>;
        };

        ////////////////////////////////////////////////////////////////////////////////
        // wide arithmetic - avoids overflow through widening

        template<typename Lhs, typename Rhs, class RepType>
        struct add_subtract_wide {
            using lean = add_subtract_lean<Lhs, Rhs, RepType>;
            using lean_rep_type = typename lean::rep_type;

            static constexpr int sufficient_sign_bits = std::is_signed<typename lean::rep_type>::value;
            static constexpr int sufficient_integer_digits = _impl::max(Lhs::integer_digits, Rhs::integer_digits);
            static constexpr int sufficient_fractional_digits = _impl::max(Lhs::fractional_digits,
                    Rhs::fractional_digits);
            static constexpr _width_type sufficient_width =
                    sufficient_sign_bits+sufficient_integer_digits+sufficient_fractional_digits;
            static constexpr int result_width = _impl::max(sufficient_width, width<lean_rep_type>::value);

            using rep_type = set_width_t<lean_rep_type, result_width>;
            using result_type = fixed_point<rep_type, -sufficient_fractional_digits>;
        };

        template<typename Lhs, typename Rhs>
        struct add_wide : add_subtract_wide<Lhs, Rhs, decltype(std::declval<typename Lhs::rep>()
                +std::declval<typename Rhs::rep>())> {
        };

        template<typename Lhs, typename Rhs>
        struct subtract_wide : add_subtract_wide<Lhs, Rhs, decltype(std::declval<typename Lhs::rep>()
                -std::declval<typename Rhs::rep>())> {
        };

        template<typename Lhs, typename Rhs>
        struct multiply_wide {
            using lean = multiply_lean<Lhs, Rhs>;
            using rep_type = typename lean::rep_type;

            static constexpr int sufficient_rep_width =
                    _impl::max(digits<rep_type>::value, Lhs::digits+Rhs::digits)+std::is_signed<rep_type>::value;

            using result_type = set_width_t<typename lean::result_type, sufficient_rep_width>;
        };

        template<typename Lhs, typename Rhs>
        struct divide_wide {
            using lean = divide_lean<Lhs, Rhs>;
            using rep_type = typename lean::rep_type;

            static constexpr int sufficient_rep_width =
                    _impl::max(digits<rep_type>::value, Lhs::digits+Rhs::digits)+std::is_signed<rep_type>::value;

            using intermediate_type = set_width_t<typename lean::intermediate_type, sufficient_rep_width>;
            using result_type = set_width_t<typename lean::result_type, sufficient_rep_width>;
        };

        ////////////////////////////////////////////////////////////////////////////////
        // aliases

        template<class BinaryPair> using add_fn = add_lean<BinaryPair>;
        template<class Lhs, class Rhs> using subtract_fn = subtract_lean<Lhs, Rhs>;
        template<class Lhs, class Rhs> using multiply_fn = multiply_lean<Lhs, Rhs>;
        template<class Lhs, class Rhs> using divide_fn = divide_lean<Lhs, Rhs>;

        template<class Lhs, class Rhs> using add_op = add_wide<Lhs, Rhs>;
        template<class Lhs, class Rhs> using subtract_op = subtract_wide<Lhs, Rhs>;
        template<class Lhs, class Rhs> using multiply_op = multiply_wide<Lhs, Rhs>;
        template<class Lhs, class Rhs> using divide_op = divide_wide<Lhs, Rhs>;
    }

    /// \brief calculates the negative of a \ref fixed_point value
    ///
    /// \param rhs input value
    ///
    /// \return negative: - rhs
    ///
    /// \note This function negates the value
    /// without performing any additional scaling or conversion.
    ///
    /// \sa add, subtract, multiply, divide

    template<class RhsRep, int RhsExponent>
    constexpr auto negate(const fixed_point<RhsRep, RhsExponent>& rhs)
    -> fixed_point<decltype(-rhs.data()), RhsExponent>
    {
        using result_type = fixed_point<decltype(-rhs.data()), RhsExponent>;
        return result_type::from_data(-rhs.data());
    }

    /// \brief calculates the sum of two \ref fixed_point values
    ///
    /// \param lhs, rhs augend and addend
    ///
    /// \return sum: lhs + rhs
    ///
    /// \note This function add the values
    /// without performing any additional scaling or conversion.
    ///
    /// \sa negate, subtract, multiply, divide

    template<class Lhs, class Rhs>
    constexpr auto add(const Lhs& lhs, const Rhs& rhs)
    -> typename _fixed_point_impl::add_fn<_fixed_point_impl::binary_pair<Lhs, Rhs>>::result_type
    {
        using result_type = typename _fixed_point_impl::add_fn<_fixed_point_impl::binary_pair<Lhs, Rhs>>::result_type;
        return result_type::from_data(static_cast<result_type>(lhs).data()+static_cast<result_type>(rhs).data());
    }

    /// \brief calculates the difference of two \ref fixed_point values
    ///
    /// \param lhs, rhs minuend and subtrahend
    ///
    /// \return difference: lhs - rhs
    ///
    /// \note This function subtracts the values
    /// without performing any additional scaling or conversion.
    ///
    /// \sa negate, add, multiply, divide

    template<class LhsRep, int LhsExponent, class RhsRep, int RhsExponent>
    constexpr auto subtract(
            const fixed_point<LhsRep, LhsExponent>& lhs,
            const fixed_point<RhsRep, RhsExponent>& rhs)
    -> typename _fixed_point_impl::subtract_fn<fixed_point<LhsRep, LhsExponent>, fixed_point<RhsRep, RhsExponent>>::result_type
    {
        using result_type = typename _fixed_point_impl::subtract_fn<fixed_point<LhsRep, LhsExponent>, fixed_point<RhsRep, RhsExponent>>::result_type;
        return result_type::from_data(static_cast<result_type>(lhs).data()-static_cast<result_type>(rhs).data());
    }

    /// \brief calculates the product of two \ref fixed_point factors
    ///
    /// \param lhs, rhs the factors
    ///
    /// \return product: lhs * rhs
    ///
    /// \note This function multiplies the values
    /// without performing any additional scaling or conversion.
    ///
    /// \sa negate, add, subtract, divide

    template<class LhsRep, int LhsExponent, class RhsRep, int RhsExponent>
    constexpr auto multiply(
            const fixed_point<LhsRep, LhsExponent>& lhs,
            const fixed_point<RhsRep, RhsExponent>& rhs)
    -> typename _fixed_point_impl::multiply_lean<fixed_point<LhsRep, LhsExponent>, fixed_point<RhsRep, RhsExponent>>::result_type
    {
        using result_type = typename _fixed_point_impl::multiply_lean<fixed_point<LhsRep, LhsExponent>, fixed_point<RhsRep, RhsExponent>>::result_type;
        using result_type_rep = typename result_type::rep;
        return result_type::from_data(
                static_cast<result_type_rep>(lhs.data())*static_cast<result_type_rep>(rhs.data()));
    }

    /// \brief calculates the quotient of two \ref fixed_point values
    ///
    /// \param lhs, rhs dividend and divisor
    ///
    /// \return quotient: lhs / rhs
    ///
    /// \note This function divides the values
    /// without performing any additional scaling or conversion.
    ///
    /// \sa negate, add, subtract, multiply

    template<class LhsRep, int LhsExponent, class RhsRep, int RhsExponent>
    constexpr auto divide(
            const fixed_point<LhsRep, LhsExponent>& lhs,
            const fixed_point<RhsRep, RhsExponent>& rhs)
    -> typename _fixed_point_impl::divide_lean<fixed_point<LhsRep, LhsExponent>, fixed_point<RhsRep, RhsExponent>>::result_type
    {
        using divide = typename _fixed_point_impl::divide_lean<fixed_point<LhsRep, LhsExponent>, fixed_point<RhsRep, RhsExponent>>;
        using intermediate_type = typename divide::intermediate_type;
        using result_type = typename divide::result_type;

        // but now result_type is wrong because we've changed what lhs is!
        return result_type::from_data(static_cast<intermediate_type>(lhs).data()/rhs.data());
    }

    ////////////////////////////////////////////////////////////////////////////////
    // (fixed_point @ fixed_point) comparison operators

    template<class Rep, int Exponent>
    constexpr bool operator==(
            const fixed_point<Rep, Exponent>& lhs,
            const fixed_point<Rep, Exponent>& rhs)
    {
        return lhs.data()==rhs.data();
    }

    template<class Rep, int Exponent>
    constexpr bool operator!=(
            const fixed_point<Rep, Exponent>& lhs,
            const fixed_point<Rep, Exponent>& rhs)
    {
        return lhs.data()!=rhs.data();
    }

    template<class Rep, int Exponent>
    constexpr bool operator<(
            const fixed_point<Rep, Exponent>& lhs,
            const fixed_point<Rep, Exponent>& rhs)
    {
        return lhs.data()<rhs.data();
    }

    template<class Rep, int Exponent>
    constexpr bool operator>(
            const fixed_point<Rep, Exponent>& lhs,
            const fixed_point<Rep, Exponent>& rhs)
    {
        return lhs.data()>rhs.data();
    }

    template<class Rep, int Exponent>
    constexpr bool operator>=(
            const fixed_point<Rep, Exponent>& lhs,
            const fixed_point<Rep, Exponent>& rhs)
    {
        return lhs.data()>=rhs.data();
    }

    template<class Rep, int Exponent>
    constexpr bool operator<=(
            const fixed_point<Rep, Exponent>& lhs,
            const fixed_point<Rep, Exponent>& rhs)
    {
        return lhs.data()<=rhs.data();
    }

    ////////////////////////////////////////////////////////////////////////////////
    // (fixed_point @ fixed_point) arithmetic operators

    // negate
    template<class RhsRep, int RhsExponent>
    constexpr auto operator-(const fixed_point<RhsRep, RhsExponent>& rhs)
    -> fixed_point<decltype(-rhs.data()), RhsExponent>
    {
        using result_type = fixed_point<decltype(-rhs.data()), RhsExponent>;
        return result_type::from_data(-rhs.data());
    }

    // add
    template<
            class LhsRep, int LhsExponent,
            class RhsRep, int RhsExponent>
    constexpr auto operator+(
            const fixed_point<LhsRep, LhsExponent>& lhs,
            const fixed_point<RhsRep, RhsExponent>& rhs)
    -> typename _fixed_point_impl::add_op<fixed_point<LhsRep, LhsExponent>, fixed_point<RhsRep, RhsExponent>>::result_type
    {
        using result_type = typename _fixed_point_impl::add_op<fixed_point<LhsRep, LhsExponent>, fixed_point<RhsRep, RhsExponent>>::result_type;
        return result_type::from_data(static_cast<result_type>(lhs).data()+static_cast<result_type>(rhs).data());
    }

    // subtract
    template<
            class LhsRep, int LhsExponent,
            class RhsRep, int RhsExponent>
    constexpr auto operator-(
            const fixed_point<LhsRep, LhsExponent>& lhs,
            const fixed_point<RhsRep, RhsExponent>& rhs)
    -> typename _fixed_point_impl::subtract_op<fixed_point<LhsRep, LhsExponent>, fixed_point<RhsRep, RhsExponent>>::result_type
    {
        using result_type = typename _fixed_point_impl::subtract_op<fixed_point<LhsRep, LhsExponent>, fixed_point<RhsRep, RhsExponent>>::result_type;
        return result_type::from_data(static_cast<result_type>(lhs).data()-static_cast<result_type>(rhs).data());
    }

    // multiply
    template<
            class LhsRep, int LhsExponent,
            class RhsRep, int RhsExponent>
    constexpr auto operator*(
            const fixed_point<LhsRep, LhsExponent>& lhs,
            const fixed_point<RhsRep, RhsExponent>& rhs)
    -> typename _fixed_point_impl::multiply_op<fixed_point<LhsRep, LhsExponent>, fixed_point<RhsRep, RhsExponent>>::result_type
    {
        using result_type = typename _fixed_point_impl::multiply_op<fixed_point<LhsRep, LhsExponent>, fixed_point<RhsRep, RhsExponent>>::result_type;
        using result_type_rep = typename result_type::rep;
        return result_type::from_data(
                static_cast<result_type_rep>(lhs.data())*static_cast<result_type_rep>(rhs.data()));
    }

    // divide
    template<class LhsRep, int LhsExponent, class RhsRep, int RhsExponent>
    constexpr auto operator/(
            const fixed_point<LhsRep, LhsExponent>& lhs,
            const fixed_point<RhsRep, RhsExponent>& rhs)
    -> typename _fixed_point_impl::divide_op<fixed_point<LhsRep, LhsExponent>, fixed_point<RhsRep, RhsExponent>>::result_type
    {
        using divide = typename _fixed_point_impl::divide_op<fixed_point<LhsRep, LhsExponent>, fixed_point<RhsRep, RhsExponent>>;
        using intermediate_type = typename divide::intermediate_type;
        using result_type = typename divide::result_type;

        // but now result_type is wrong because we've changed what lhs is!
        return result_type::from_data(static_cast<intermediate_type>(lhs).data()/rhs.data());
    }

    ////////////////////////////////////////////////////////////////////////////////
    // heterogeneous operator overloads
    //
    // compare two objects of different fixed_point specializations

    template<class Lhs, class Rhs>
    constexpr auto operator==(const Lhs& lhs, const Rhs& rhs)
    -> typename std::enable_if<
            _fixed_point_impl::is_fixed_point<Lhs>::value || _fixed_point_impl::is_fixed_point<Rhs>::value, bool>::type
    {
        using common_type = _impl::common_type_t<Lhs, Rhs>;
        return static_cast<common_type>(lhs)==static_cast<common_type>(rhs);
    }

    template<class Lhs, class Rhs>
    constexpr auto operator!=(const Lhs& lhs, const Rhs& rhs)
    -> typename std::enable_if<
            _fixed_point_impl::is_fixed_point<Lhs>::value || _fixed_point_impl::is_fixed_point<Rhs>::value, bool>::type
    {
        using common_type = _impl::common_type_t<Lhs, Rhs>;
        return static_cast<common_type>(lhs)!=static_cast<common_type>(rhs);
    }

    template<class Lhs, class Rhs>
    constexpr auto operator<(const Lhs& lhs, const Rhs& rhs)
    -> typename std::enable_if<
            _fixed_point_impl::is_fixed_point<Lhs>::value || _fixed_point_impl::is_fixed_point<Rhs>::value, bool>::type
    {
        using common_type = _impl::common_type_t<Lhs, Rhs>;
        return static_cast<common_type>(lhs)<static_cast<common_type>(rhs);
    }

    template<class Lhs, class Rhs>
    constexpr auto operator>(const Lhs& lhs, const Rhs& rhs)
    -> typename std::enable_if<
            _fixed_point_impl::is_fixed_point<Lhs>::value || _fixed_point_impl::is_fixed_point<Rhs>::value, bool>::type
    {
        using common_type = _impl::common_type_t<Lhs, Rhs>;
        return static_cast<common_type>(lhs)>static_cast<common_type>(rhs);
    }

    template<class Lhs, class Rhs>
    constexpr auto operator>=(const Lhs& lhs, const Rhs& rhs)
    -> typename std::enable_if<
            _fixed_point_impl::is_fixed_point<Lhs>::value || _fixed_point_impl::is_fixed_point<Rhs>::value, bool>::type
    {
        using common_type = _impl::common_type_t<Lhs, Rhs>;
        return static_cast<common_type>(lhs)>=static_cast<common_type>(rhs);
    }

    template<class Lhs, class Rhs>
    constexpr auto operator<=(const Lhs& lhs, const Rhs& rhs)
    -> typename std::enable_if<
            _fixed_point_impl::is_fixed_point<Lhs>::value || _fixed_point_impl::is_fixed_point<Rhs>::value, bool>::type
    {
        using common_type = _impl::common_type_t<Lhs, Rhs>;
        return static_cast<common_type>(lhs)<=static_cast<common_type>(rhs);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // (fixed_point @ non-fixed_point) arithmetic operators

    // fixed-point, integer -> fixed-point
    template<
            class LhsRep, int LhsExponent,
            class RhsInteger,
            typename = typename std::enable_if<is_integral<RhsInteger>::value>::type>
    constexpr auto operator+(const fixed_point<LhsRep, LhsExponent>& lhs, const RhsInteger& rhs)
    -> decltype(lhs + fixed_point<RhsInteger, 0>{rhs})
    {
        return lhs + fixed_point<RhsInteger, 0>{rhs};
    }

    template<
            class LhsRep, int LhsExponent,
            class RhsInteger,
            typename = typename std::enable_if<is_integral<RhsInteger>::value>::type>
    constexpr auto operator-(const fixed_point<LhsRep, LhsExponent>& lhs, const RhsInteger& rhs)
    -> decltype(lhs - fixed_point<RhsInteger, 0>{rhs})
    {
        return lhs - fixed_point<RhsInteger, 0>{rhs};
    }

    template<
            class LhsRep, int LhsExponent,
            class RhsInteger,
            typename = typename std::enable_if<is_integral<RhsInteger>::value>::type>
    constexpr auto operator*(const fixed_point<LhsRep, LhsExponent>& lhs, const RhsInteger& rhs)
    -> decltype(lhs*fixed_point<RhsInteger>(rhs))
    {
        return lhs*fixed_point<RhsInteger>(rhs);
    }

    template<
            class LhsRep, int LhsExponent,
            class RhsInteger,
            typename = typename std::enable_if<is_integral<RhsInteger>::value>::type>
    constexpr auto operator/(const fixed_point<LhsRep, LhsExponent>& lhs, const RhsInteger& rhs)
    -> decltype(lhs/fixed_point<RhsInteger>{rhs})
    {
        return lhs/fixed_point<RhsInteger>{rhs};
    }

    // integer. fixed-point -> fixed-point
    template<
            class LhsInteger,
            class RhsRep, int RhsExponent,
            typename = typename std::enable_if<is_integral<LhsInteger>::value>::type>
    constexpr auto operator+(const LhsInteger& lhs, const fixed_point<RhsRep, RhsExponent>& rhs)
    -> decltype(fixed_point<LhsInteger, 0>{lhs} + rhs)
    {
        return fixed_point<LhsInteger, 0>{lhs} + rhs;
    }

    template<
            class LhsInteger,
            class RhsRep, int RhsExponent,
            typename = typename std::enable_if<is_integral<LhsInteger>::value>::type>
    constexpr auto operator-(const LhsInteger& lhs, const fixed_point<RhsRep, RhsExponent>& rhs)
    -> decltype(fixed_point<LhsInteger>{lhs}-rhs)
    {
        return fixed_point<LhsInteger>{lhs}-rhs;
    }

    template<
            class LhsInteger,
            class RhsRep, int RhsExponent,
            typename = typename std::enable_if<is_integral<LhsInteger>::value>::type>
    constexpr auto operator*(const LhsInteger& lhs, const fixed_point<RhsRep, RhsExponent>& rhs)
    -> decltype(fixed_point<LhsInteger>{lhs}*rhs)
    {
        return fixed_point<LhsInteger>{lhs}*rhs;
    }

    template<
            class LhsInteger,
            class RhsRep, int RhsExponent,
            typename = typename std::enable_if<is_integral<LhsInteger>::value>::type>
    constexpr auto operator/(const LhsInteger& lhs, const fixed_point<RhsRep, RhsExponent>& rhs)
    -> decltype(fixed_point<LhsInteger>{lhs}/rhs)
    {
        return fixed_point<LhsInteger>{lhs}/rhs;
    }

    // fixed-point, floating-point -> floating-point
    template<class LhsRep, int LhsExponent, class RhsFloat, typename = typename std::enable_if<std::is_floating_point<RhsFloat>::value>::type>
    constexpr auto operator+(const fixed_point<LhsRep, LhsExponent>& lhs, const RhsFloat& rhs)-> _impl::common_type_t<fixed_point<LhsRep, LhsExponent>, RhsFloat>
    {
        using result_type = _impl::common_type_t<fixed_point<LhsRep, LhsExponent>, RhsFloat>;
        return static_cast<result_type>(lhs)+static_cast<result_type>(rhs);
    }

    template<class LhsRep, int LhsExponent, class RhsFloat, typename = typename std::enable_if<std::is_floating_point<RhsFloat>::value>::type>
    constexpr auto operator-(const fixed_point<LhsRep, LhsExponent>& lhs, const RhsFloat& rhs)-> _impl::common_type_t<fixed_point<LhsRep, LhsExponent>, RhsFloat>
    {
        using result_type = _impl::common_type_t<fixed_point<LhsRep, LhsExponent>, RhsFloat>;
        return static_cast<result_type>(lhs)-static_cast<result_type>(rhs);
    }

    template<class LhsRep, int LhsExponent, class RhsFloat>
    constexpr auto operator*(
            const fixed_point<LhsRep, LhsExponent>& lhs,
            const RhsFloat& rhs)
    -> _impl::common_type_t<
            fixed_point<LhsRep, LhsExponent>,
            typename std::enable_if<std::is_floating_point<RhsFloat>::value, RhsFloat>::type>
    {
        using result_type = _impl::common_type_t<fixed_point<LhsRep, LhsExponent>, RhsFloat>;
        return static_cast<result_type>(lhs)*rhs;
    }

    template<class LhsRep, int LhsExponent, class RhsFloat>
    constexpr auto operator/(
            const fixed_point<LhsRep, LhsExponent>& lhs,
            const RhsFloat& rhs)
    -> _impl::common_type_t<
            fixed_point<LhsRep, LhsExponent>,
            typename std::enable_if<std::is_floating_point<RhsFloat>::value, RhsFloat>::type>
    {
        using result_type = _impl::common_type_t<fixed_point<LhsRep, LhsExponent>, RhsFloat>;
        return static_cast<result_type>(lhs)/rhs;
    }

    // floating-point, fixed-point -> floating-point
    template<class LhsFloat, class RhsRep, int RhsExponent, typename = typename std::enable_if<std::is_floating_point<LhsFloat>::value>::type>
    constexpr auto operator+(const LhsFloat& lhs, const fixed_point<RhsRep, RhsExponent>& rhs)-> _impl::common_type_t<LhsFloat, fixed_point<RhsRep, RhsExponent>>
    {
        using result_type = _impl::common_type_t<LhsFloat, fixed_point<RhsRep, RhsExponent>>;
        return static_cast<result_type>(lhs)+static_cast<result_type>(rhs);
    }

    template<class LhsFloat, class RhsRep, int RhsExponent, typename = typename std::enable_if<std::is_floating_point<LhsFloat>::value>::type>
    constexpr auto operator-(const LhsFloat& lhs, const fixed_point<RhsRep, RhsExponent>& rhs)-> _impl::common_type_t<LhsFloat, fixed_point<RhsRep, RhsExponent>>
    {
        using result_type = _impl::common_type_t<LhsFloat, fixed_point<RhsRep, RhsExponent>>;
        return static_cast<result_type>(lhs)-static_cast<result_type>(rhs);
    }

    template<class LhsFloat, class RhsRep, int RhsExponent>
    constexpr auto operator*(
            const LhsFloat& lhs,
            const fixed_point<RhsRep, RhsExponent>& rhs)
    -> _impl::common_type_t<
            typename std::enable_if<std::is_floating_point<LhsFloat>::value, LhsFloat>::type,
            fixed_point<RhsRep, RhsExponent>>
    {
        using result_type = _impl::common_type_t<fixed_point<RhsRep, RhsExponent>, LhsFloat>;
        return lhs*static_cast<result_type>(rhs);
    }

    template<class LhsFloat, class RhsRep, int RhsExponent>
    constexpr auto operator/(
            const LhsFloat& lhs,
            const fixed_point<RhsRep, RhsExponent>& rhs)
    -> _impl::common_type_t<
            typename std::enable_if<std::is_floating_point<LhsFloat>::value, LhsFloat>::type,
            fixed_point<RhsRep, RhsExponent>>
    {
        using result_type = _impl::common_type_t<fixed_point<RhsRep, RhsExponent>, LhsFloat>;
        return lhs/
                static_cast<result_type>(rhs);
    }

    template<class LhsRep, int Exponent, class Rhs>
    fixed_point<LhsRep, Exponent>& operator+=(fixed_point<LhsRep, Exponent>& lhs, const Rhs& rhs)
    {
        return lhs = lhs+fixed_point<LhsRep, Exponent>(rhs);
    }

    template<class LhsRep, int Exponent, class Rhs>
    fixed_point<LhsRep, Exponent>& operator-=(fixed_point<LhsRep, Exponent>& lhs, const Rhs& rhs)
    {
        return lhs = lhs-fixed_point<LhsRep, Exponent>(rhs);
    }

    template<class LhsRep, int Exponent>
    template<class Rhs>
    fixed_point<LhsRep, Exponent>&
    fixed_point<LhsRep, Exponent>::operator*=(const Rhs& rhs)
    {
        _r *= static_cast<rep>(rhs);
        return *this;
    }

    template<class LhsRep, int Exponent>
    template<class Rhs>
    fixed_point<LhsRep, Exponent>&
    fixed_point<LhsRep, Exponent>::operator/=(const Rhs& rhs)
    {
        _r /= static_cast<rep>(rhs);
        return *this;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // shift operators

    template<typename LhsRep, int LhsExponent, typename Rhs>
    constexpr fixed_point<LhsRep, LhsExponent>
    operator<<(const fixed_point<LhsRep, LhsExponent>& lhs, const Rhs& rhs)
    {
        return fixed_point<LhsRep, LhsExponent>::from_data(lhs.data() << rhs);
    };

    template<typename LhsRep, int LhsExponent, typename Rhs>
    constexpr fixed_point<LhsRep, LhsExponent>
    operator>>(const fixed_point<LhsRep, LhsExponent>& lhs, const Rhs& rhs)
    {
        return fixed_point<LhsRep, LhsExponent>::from_data(lhs.data() >> rhs);
    };
}

#include "bits/fixed_point_extras.h"

#endif	// SG14_FIXED_POINT_H
