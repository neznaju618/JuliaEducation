module LinearAlgebraMethods

    export norm, norm2, norm2square, diag, solve, LUdecomposition, LUsolve, ortogonalize, 
    QRdecomposition, QReigenvalues, LUinverse, inversebyGauss

    function norm(v::Vector{<:Real})
        return maximum(abs.(v))        
    end

    function norm(M::Matrix{<:Real})
        return norm((abs.(M)) * ones(size(M, 2)))
    end

    function norm2square(v::Vector{<:Real})
        return sum(abs2.(v))
    end

    function norm2(v::Vector{<:Real})
        return sqrt(sum(abs2.(v)))
    end

    function diag(A::Matrix{<:Real})
        return [A[i, i] for i in 1: min(size(A, 1), size(A, 2))]
    end

    function solve(A::Matrix{<:Real}, b::Vector{<:Real}; eps=1e-15)
        if size(A, 1) != size(A, 2) || size(A, 1) != length(b)
            throw(ArgumentError("Incorrect data sizes"))
        end
        result = zeros(length(b))
        while norm(A*result .- b) >= eps
            for i in 1:length(b)
                result[i] = (-A[i, :]' * result + b[i]) / A[i, i] + result[i]
            end
        end
        return result
    end

    function inversebyGauss(A::Matrix{<:Real})
        if size(A, 1) != size(A, 2)
            throw(ArgumentError("Matrix is not square"))
        end
        result = [float(i == j) for i in 1:size(A, 1), j in 1:size(A, 2)]
        M = copy(A)
        function findrow(currentcol::Integer)
            for k in currentcol + 1:size(M, 2)
                if M[k, currentcol] != 0
                    return k
                end
            end
            throw(ArgumentError("Singular matrix"))
        end
        for j in 1:size(M, 1) - 1 
            for i in j+1:size(M, 1)
                if M[j, j] == 0
                    k = findrow(j)
                    M[k, :], M[j, :] = M[j, :], M[k, :]
                    result[k, :], result[j, :] = result[j, :], result[k, :]
                end
                if i != j
                    M[i, :] .-= M[i, j] / M[j, j] .* M[j, :]
                    result[i, :] -= result[i, j] / M[j, j] .* result[j, :]
                end
            end
        end
        for j in size(M, 1):-1:1
            for i in j:-1:1
                if i != j
                    M[i, :] .-= M[i, j] / M[j, j] .* M[j, :]
                    result[i, :] -= result[i, j] / M[j, j] .* result[j, :]
                else
                    M[i, :] ./= M[j, j]
                    result[i, :] ./= M[j, j]
                end
            end
        end
        return result
    end

    function LUiteration(A::Matrix{<:Real})
        N = size(A, 1)
        w = A[1, 2:N]
        v = A[2:N, 1] ./ A[1, 1]
        Anext = A[2:N, 2:N] .- v * w'
        return w, v, Anext
    end

    function LUdecomposition(A::Matrix{<:Real})
        N = size(A, 1)
        if N != size(A, 2) throw(ArgumentError("Matrix is not square")) end
        L = zeros(N, N)
        U = zeros(N, N)
        Anext = copy(A)
        for i in 1:N-1
            L[i, i] = 1
            U[i, i] = Anext[1, 1]
            if U[i, i] == 0 throw(ArgumentError("Singular matrix")) end
            w, v, Anext = LUiteration(Anext)
            L[i+1:N, i] = v
            U[i, i+1:N] = w
        end
        L[N, N] = 1
        U[N, N] = Anext[1, 1]
        if U[N, N] == 0 throw(ArgumentError("Singular matrix")) end
        return L, U
    end

    function solveLTsystem(L::Matrix{<:Real}, b::Vector{<:Real})
        if size(L, 1) != size(L, 2) throw(ArgumentError("Matrix is not square")) end
        result = zeros(size(L, 1))
        for i in 1:size(L, 1)
            if L[i, i] == 0 throw(ArgumentError("Singular matrix")) end
            result[i] = (b[i] - L[i, :]' * result) / L[i, i]
        end
        return result
    end

    function solveUTsystem(U::Matrix{<:Real}, b::Vector{<:Real})
        if size(U, 1) != size(U, 2) throw(ArgumentError("Matrix is not square")) end
        result = zeros(size(U, 1))
        for i in size(U, 1):-1:1
            if U[i, i] == 0 throw(ArgumentError("Singular matrix")) end
            result[i] = (b[i] - U[i, :]' * result) / U[i, i]
        end
        return result
    end

    function LUsolve(A::Matrix{<:Real}, b::Vector{<:Real})
        L, U = LUdecomposition(A)
        return solveUTsystem(U, solveLTsystem(L, b))
    end
    
    function ortogonalize(A::Matrix{<:Real})
        function proj(a::Vector{<:Real}, b::Vector{<:Real})
            return a'*b .* b
        end
        result = zeros(size(A))
        result[:, 1] = A[:, 1] ./ norm2(A[:, 1])
        result[:, 2] = A[:, 2] .- proj(A[:, 2], result[:, 1])
        result[:, 2] ./= norm2(result[:, 2])
        for j in 3:size(A, 2)
            a = A[:, j] .- proj(A[:, j], result[:, 1])
            for i in 2:j - 2
                a = a .- proj(a, result[:, i])
            end
            result[:, j] = a .- proj(a, result[:, j-1])
            result[:, j] ./= norm2(result[:, j])
        end
        return result
    end

    function QRdecomposition(A::Matrix{<:Real})
        if size(A, 2) > size(A, 1) throw(ArgumentError("Incorrect mstrix sizes")) end
        Q = ortogonalize(A)
        R = Q' * A
        return Q, R        
    end

    function QReigenvalues(A::Matrix{<:Real}; ϵ=4e-16)
        if size(A, 1) != size(A, 2) throw(ArgumentError("Matrix is not square")) end
        Acurrent = A
        error = Acurrent .+ 1
        i = 0
        while norm(error) >= ϵ
            Qk, Rk = QRdecomposition(Acurrent)
            error = Acurrent .- Qk * Rk
            Acurrent = Rk * Qk
            i += 1
        end
        return diag(Acurrent) 
    end

    function inverseLT(L::Matrix{<:Real})
        if size(L, 1) != size(L, 2) throw(ArgumentError("Matrix is not square")) end
        result = zeros(size(L))
        for i in 1:size(L, 1), j in 1:i
            if i == j 
                result[i, j] = 1 / L[i, j]
            else
                result[i, j] = -L[i, :]' * result[:, j] ./ L[i, i]
            end
        end 
        return result
    end

    function inverseUT(U::Matrix{<:Real})
        if size(U, 1) != size(U, 2) throw(ArgumentError("Matrix is not square")) end
        result = zeros(size(U))
        for j in 1:size(U, 2), i in 1:j
            if i == j 
                result[i, j] = 1 / U[i, j]
            else
                result[i, j] = -U[:, j]' * result[i, :] ./ U[j, j]
            end
        end
        return result
    end
    
    function LUinverse(A::Matrix{<:Real})
        if size(A, 1) != size(A, 2) throw(ArgumentError("Matrix is not square")) end
        L, U = LUdecomposition(A)
        return inverseUT(U) * inverseLT(L)
    end
    
end
