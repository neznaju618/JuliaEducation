module DiffEquations

    export DiffEquationSystem, LinearDiffEquationWithConstCoefs, 
    LinearDiffEquationWithVarCoefs, solve

    struct VectorFunction
        functions::Vector{<:Function}
    end

    function compute(self::VectorFunction, t::Real, y::Vector{<:Real})
        return map(f::Function -> f(t, y), self.functions)
    end

    function compute(self::VectorFunction, t::Real)
        return map(f::Function -> f(t), self.functions)
    end

    struct DiffEquationSystem
        F::VectorFunction
        initcond::Vector{<:Real}
        start::Real
    end

    function solve(self::DiffEquationSystem, finish::Real, N::Integer)
        h = (finish - self.start) / N
        current_t = self.start
        result = zeros(length(self.initcond), N + 1)
        result[:, 1] = copy(self.initcond)
        for n in 2:N + 1
            forecast = result[:, n-1] + h * compute(self.F, current_t, result[:, n-1])
            result[:, n] = result[:, n-1] + h * (compute(self.F, current_t, result[:, n-1]) + compute(self.F, current_t += h, forecast)) / 2
        end
        return result
    end

    abstract type LinearDiffEquation end

    struct LinearDiffEquationWithConstCoefs <: LinearDiffEquation
        coefs::Vector{<:Real}
        f::Function
        initcond::Vector{<:Real}
        start::Real
    end

    function solve(self::LinearDiffEquationWithConstCoefs, finish::Real, N::Integer)
        F = [(i != length(self.initcond)) ? (t::Real, y::Vector{<:Real}) -> y[i+1] : (t::Real, y::Vector{<:Real}) -> (-self.coefs[1:end-1]' * y + self.f(t)) / self.coefs[end] for i in 1:length(self.initcond)]
        system = DiffEquationSystem(VectorFunction(F), self.initcond, self.start)
        return solve(system, finish, N)
    end
    
    struct LinearDiffEquationWithVarCoefs <:LinearDiffEquation
        coefs::VectorFunction
        f::Function
        initcond::Vector{<:Real}
        start::Real
    end

    function LinearDiffEquationWithVarCoefs(coefs::Vector{<:Function}, f::Function, initcond::Vector{<:Real}, start::Real)
        return LinearDiffEquationWithVarCoefs(VectorFunction(coefs), f, initcond, start)
    end

    function solve(self::LinearDiffEquationWithVarCoefs, finish::Real, N::Integer)
        F = [(i == length(self.initcond)) ? (t::Real, y::Vector{<:Real}) -> y[i+1] : (t::Real, y::Vector{<:Real}) -> (-compute(self.coefs, t)[1:end-1]' * y + self.f(t, y)) / self.coefs.functions[end](t) for i in 1:length(self.initcond)] 
        system = DiffEquationSystem(VectorFunction(F), self.initcond, self.start)
        return solve(system, finish, N)
    end

end