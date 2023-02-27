module OneDimIntegrations
    using LinearAlgebra
    export integrate_by_simpson, integrate_by_trapez, itegrate, IntegralEquation, 
    FredholmIntegralEquation, solve_by_simpson, solve_by_trapez, solve                                                                                              

    function integrate_by_trapez(f::Function, start::Real, finish::Real, N::Integer) 
        result = 0.0
        h = (finish - start) / N
        current_x = start
        while current_x < finish - h 
            result += (f(current_x) + f(current_x += h)) * h / 2
        end
        return result
    end

    function integrate_by_simpson(f::Function, start::Real, finish::Real, N::Integer)
        result = 0.0
        h = (finish - start) / N
        current_x = start
        while current_x < finish - h
            result += f(current_x) + 4 * f(current_x += h) + f(current_x += h)
        end
        return result * h / 3
    end

    
    function integrate(f::Function, start::Real, finish::Real, N::Integer)
        mod(N, 2) == 0 ? integrate_by_simpson(f, start, finish, N) : integrate_by_trapez(f, start, finish, N)
    end

    abstract type IntegralEquation end

    struct FredholmIntegralEquation <: IntegralEquation
        λ:: Real
        core::Function
        rightside::Function
        start::Real
        finish::Real
    end

    function solve_by_trapez(self::FredholmIntegralEquation, N::Integer; ϵ = 1e-15)
        h = (self.finish - self.start) / N
        grid = collect(LinRange(self.start, self.finish, N + 1))
        K = (self.λ * h) .* [self.core(s, t) for s in grid, t in grid]
        K[:, 1] ./= 2
        K[:, end] ./= 2
        b = map(self.rightside, grid)
        result = copy(b)
        while maximum(abs.(K*result .- result .+ b)) > ϵ
            result = K*result + b
        end
        return result
    end

    function solve_by_simpson(self::FredholmIntegralEquation, N::Integer; ϵ = 1e-15)
        h = (self.finish - self.start) / N
        grid = collect(LinRange(self.start, self.finish, N + 1))
        K = (self.λ * h / 3) .* [self.core(s, t) * 2^(1 + mod(1+idx, 2)) for s in grid, (idx,t) in enumerate(grid)]
        K[:, 1] ./= 2
        K[:, end] ./= 2
        b = map(self.rightside, grid)
        result = copy(b)
        while maximum(abs.(K*result .- result .+ b)) > ϵ
            result = K*result + b
        end
        return result
    end

    function solve(self::FredholmIntegralEquation, N::Integer; ϵ = 1e-15)
        mod(N, 2) == 0 ? solve_by_simpson(self, N, ϵ=ϵ) : solve_by_trapez(self, N, ϵ=ϵ)
    end

    struct VolterraIntegralEquation <: IntegralEquation
        λ:: Real
        core::Function
        rightside::Function
        start
    end

    function solve(self::VolterraIntegralEquation, finish::Real, N::Integer)
        if finish < self.start
            throw(ArgumentError("Incorrect finish point"))
        end
        h = (finish - self.start) / N
        grid = collect(LinRange(self.start, finish, N + 1))
        K = -(self.λ * h) .* [(j ≤ i) * self.core(s, t) for (i, s) in enumerate(grid), (j, t) in enumerate(grid)]
        K[:, 1] ./= 2
        K[:, end] ./= 2
        b = map(self.rightside, grid)
        K += UniformScaling(1)
        result = zeros(N+1)
        for i in 1:N+1
            result[i] = (b[i] - K[i, :]' * result) / K[i, i]
        end
        return result
    end

end
