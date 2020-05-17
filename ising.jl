using PyPlot, StatsBase, LsqFit

#periodic boundary conditions
function Periodic(shift,size)
    if shift == 0
        shift = size
    elseif shift == size+1
        shift = 1
    end
    return shift
end

#calculate the energy
function energy(S)
    sum(- S .* (circshift(S, (1, 0)) .+ circshift(S, (0, 1))))
end

#metropolis step
function metropolis_step!(S,T)
    N = size(S,1)
    r_x = rand(1:N)
    r_y = rand(1:N)
    delta_E = 2 * S[r_x,r_y] .* (circshift(S, (0, 1)) .+
                   circshift(S, (0, -1)) .+
                   circshift(S, (1, 0)) .+
                   circshift(S, (-1, 0)))
    if delta_E[r_x,r_y] <= 0 || rand() < exp(- (1/(T)) * delta_E[r_x,r_y])
        S[r_x,r_y] = - S[r_x,r_y]
    end
end

#wolf step
function wolff_step!(S,T)
    grow_probability = 1 - exp(-2/T)
    rand_x,rand_y = rand(1:size(S,1)),rand(1:size(S,1))
    stack = [[rand_x,rand_y]]
    value_spin = copy(S[rand_x,rand_y])
    S[rand_x,rand_y] *= -1
    while !isempty(stack)
        x,y = pop!(stack)
        above = [x,Periodic(y+1,size(S,1))]
        below = [x,Periodic(y-1,size(S,1))]
        left = [Periodic(x-1,size(S,1)),y]
        right = [Periodic(x+1,size(S,1)),y]
        neighbors = [above, below, left, right]
        for i in neighbors
            if S[i[1],i[2]] == value_spin
                if rand() < grow_probability
                    S[i[1],i[2]] *= -1
                    push!(stack,i)
                else
                end
            end
        end
    end
end

#get statistics
function energy_stats(S, T, samples, modus)
    Original = copy(S)
    energy_config = 0
    magnetization_config = 0
    energies_therma = Float64[]
    magnies_therma = Float64[]
    energies = Float64[]
    magnies = Float64[]
    autocorr = Float64[]
    for sample in 1:samples
        if modus == 1
            metropolis_step!(S,T)
        elseif modus == 2
            wolff_step!(S,T)
        end
        energy_config = energy(S)
        magnetization_config = sum(S)
        push!(energies_therma, energy_config)
        push!(magnies_therma, magnetization_config)
        if sample > samples/2
            push!(energies, energy_config)
            push!(magnies, magnetization_config)
        end
    end
    meanE = mean(energies)
    varEvec = (energies - meanE).^2
    varE = mean(varEvec)
    meanM = mean(magnies)
    varM = mean((i - meanM)^2 for i in magnies)
    autocorr = (autocor(magnies, range(1,1000); demean=true))
    erg = Dict("meanE"=>meanE / length(S),"varE"=>varE / length(S),"energies"=>energies_therma,"magnies"=>magnies_therma,"meanM"=>meanM / length(S),"varM"=>varM / length(S),"autocorr"=>autocorr, "varEvec"=> varEvec, "Evec" => energies )
    return erg
end

#sweep trough and collect statistics
function sweep(N, Trange, samples, modus)
    S = rand([1.0, -1.0], (N, N))
    meanEhist = Float64[]
    varEhist = Float64[]
    meanMhist = Float64[]
    varMhist = Float64[]
    Cvhist = Float64[]
    Sushist = Float64[]
    autocorr_hist = Float64[]

    for (i, T) in enumerate(Trange)
        erg = energy_stats(S, T, samples, modus)
        push!(meanEhist, erg["meanE"])
        push!(varEhist, erg["varE"])
        push!(meanMhist, erg["meanM"])
        push!(varMhist, erg["varM"])
        push!(autocorr_hist, sum(erg["autocorr"]))
    end

    Cvhist = varEhist ./ Trange.^2
    Sushist = varMhist ./ Trange

    erg = Dict("meanEhist"=>meanEhist,"varEhist"=>varEhist,"Cvhist"=>Cvhist,"Sushist"=>Sushist,"autohist"=>autocorr_hist)
    return erg
end

function plot_scale_e(T_array,sizes,sweeps,modus)
    index = 1
    for size in sizes
        stats_hist = sweep(size, T_array, sweeps[index], modus)
        scatter(T_array,stats_hist["meanEhist"],label = "$size x $size")
        index +=1
    end

    legend()
    xlabel("Temperatur T")
    ylabel("Energie pro Spin")
    show()
end

function plot_scale_cv(T_array,sizes,sweeps,modus)
    index = 1
    for size in sizes
        stats_hist = sweep(size, T_array, sweeps[index],modus)
        scatter(T_array,stats_hist["Cvhist"],label = "$size x $size")
        index +=1
    end

    legend()
    xlabel("Temperatur T")
    ylabel("Wärmekapazität Cv pro Spin")
end

function plot_integrated_auto(T_array,samples,modus)
    size_1 = 16
    stats_hist_1 = sweep(size_1, T_array, samples, modus)
    if modus == 1
        scatter(T_array,stats_hist_1["autohist"],label = "Metropolis")
    elseif modus == 2
        scatter(T_array,stats_hist_1["autohist"],label = "Wolff")
    end
    legend()
    xlabel("Temperatur T")
    ylabel("integrated autocorrelation time")
end

function plot_time_series_images_metro(T)
    N = 50
    samples = 1e6
    S = rand([1.0, -1.0], (N, N))
    for i in 1:samples
        metropolis_step!(S, T)
        if i == samples/80
            subplot(4,3,1)
            title("t = $(@sprintf("%d", i))")
            imshow(S, vmin=-1, vmax=1, cmap="Greys")
        end
        if i == samples/40
            subplot(4,3,2)
            title("t = $(@sprintf("%d", i))")
            imshow(S, vmin=-1, vmax=1, cmap="Greys")
        end
        if i == samples/32
            subplot(4,3,3)
            title("t = $(@sprintf("%d", i))")
            imshow(S, vmin=-1, vmax=1, cmap="Greys")
        end
        if i == samples/25
            subplot(4,3,4)
            title("t = $(@sprintf("%d", i))")
            imshow(S, vmin=-1, vmax=1, cmap="Greys")
        end
        if i == samples/20
            subplot(4,3,5)
            title("t = $(@sprintf("%d", i))")
            imshow(S, vmin=-1, vmax=1, cmap="Greys")
        end
        if i == samples/16
            subplot(4,3,6)
            title("t = $(@sprintf("%d", i))")
            imshow(S, vmin=-1, vmax=1, cmap="Greys")
        end
        if i == samples/10
            subplot(4,3,7)
            title("t = $(@sprintf("%d", i))")
            imshow(S, vmin=-1, vmax=1, cmap="Greys")
        end
        if i == samples/8
            subplot(4,3,8)
            title("t = $(@sprintf("%d", i))")
            imshow(S, vmin=-1, vmax=1, cmap="Greys")
        end
        if i == samples/5
            subplot(4,3,9)
            title("t = $(@sprintf("%d", i))")
            imshow(S, vmin=-1, vmax=1, cmap="Greys")
        end
        if i == samples/4
            subplot(4,3,10)
            title("t = $(@sprintf("%d", i))")
            imshow(S, vmin=-1, vmax=1, cmap="Greys")
        end
        if i == samples/2
            subplot(4,3,11)
            title("t = $(@sprintf("%d", i))")
            imshow(S, vmin=-1, vmax=1, cmap="Greys")
        end
        if i == samples
            subplot(4,3,12)
            title("t = $(@sprintf("%d", i))")
            imshow(S, vmin=-1, vmax=1, cmap="Greys")
        end
    end
    tight_layout()
    show()
end

function plot_time_series_images_wolff(T)
    N = 50
    samples = 24
    S = rand([1.0, -1.0], (N, N))
    for i in 1:samples
        wolff_step!(S,T)
        if i == 2
            subplot(4,3,1)
            title("t = $(@sprintf("%d", i))")
            imshow(S, vmin=-1, vmax=1, cmap="Greys")
        end
        if i == 4
            subplot(4,3,2)
            title("t = $(@sprintf("%d", i))")
            imshow(S, vmin=-1, vmax=1, cmap="Greys")
        end
        if i == 6
            subplot(4,3,3)
            title("t = $(@sprintf("%d", i))")
            imshow(S, vmin=-1, vmax=1, cmap="Greys")
        end
        if i == 8
            subplot(4,3,4)
            title("t = $(@sprintf("%d", i))")
            imshow(S, vmin=-1, vmax=1, cmap="Greys")
        end
        if i == 10
            subplot(4,3,5)
            title("t = $(@sprintf("%d", i))")
            imshow(S, vmin=-1, vmax=1, cmap="Greys")
        end
        if i == 12
            subplot(4,3,6)
            title("t = $(@sprintf("%d", i))")
            imshow(S, vmin=-1, vmax=1, cmap="Greys")
        end
        if i == 14
            subplot(4,3,7)
            title("t = $(@sprintf("%d", i))")
            imshow(S, vmin=-1, vmax=1, cmap="Greys")
        end
        if i == 16
            subplot(4,3,8)
            title("t = $(@sprintf("%d", i))")
            imshow(S, vmin=-1, vmax=1, cmap="Greys")
        end
        if i == 18
            subplot(4,3,9)
            title("t = $(@sprintf("%d", i))")
            imshow(S, vmin=-1, vmax=1, cmap="Greys")
        end
        if i == 20
            subplot(4,3,10)
            title("t = $(@sprintf("%d", i))")
            imshow(S, vmin=-1, vmax=1, cmap="Greys")
        end
        if i == 22
            subplot(4,3,11)
            title("t = $(@sprintf("%d", i))")
            imshow(S, vmin=-1, vmax=1, cmap="Greys")
        end
        if i == 24
            subplot(4,3,12)
            title("t = $(@sprintf("%d", i))")
            imshow(S, vmin=-1, vmax=1, cmap="Greys")
        end
    end
    tight_layout()
    show()
end

function thermalization(T,Ns,samples,modus)
    for N in Ns
        S = rand([1.0, -1.0], (N, N))
        thermo = energy_stats(S, T, samples, modus)
        plot(abs.(thermo["energies"]/(N*N)),label = "$N x $N")
    end
        legend()
        xlabel("sweeps")
        ylabel("energy per spin")
end

function plot_auto(samples, modus)
    N = 8
    T = 1.0
    S = rand([1.0, -1.0], (N, N))
    thermo = energy_stats(S, T, samples, modus)
    plot(thermo["autocorr"],label = "$T")
    T = 1.7
    S = rand([1.0, -1.0], (N, N))
    thermo = energy_stats(S, T, samples, modus)
    plot(thermo["autocorr"],label = "$T")
    T = 2.2
    S = rand([1.0, -1.0], (N, N))
    thermo = energy_stats(S, T, samples, modus)
    plot(thermo["autocorr"],linestyle = "--",label = "$T")
    T = 2.3
    S = rand([1.0, -1.0], (N, N))
    thermo = energy_stats(S, T, samples, modus)
    plot(thermo["autocorr"],linestyle = "--",label = "$T")
    T = 2.4
    S = rand([1.0, -1.0], (N, N))
    thermo = energy_stats(S, T, samples, modus)
    plot(thermo["autocorr"],linestyle = "--",label = "$T")
    T = 3.0
    S = rand([1.0, -1.0], (N, N))
    thermo = energy_stats(S, T, samples, modus)
    plot(thermo["autocorr"],linestyle = "-.",label = "$T")
    T = 4.0
    S = rand([1.0, -1.0], (N, N))
    thermo = energy_stats(S, T, samples, modus)
    plot(thermo["autocorr"],label = "$T")
    legend()
    xlabel("sweeps")
    ylabel("autocorrelation")
end

function plot_rew_hat(N,T,samples,modus,lab)
    S = rand([1.0, -1.0], (N, N))
    stats = energy_stats(S, T, samples, modus)
    varEvec = stats["varEvec"]
    Evec = stats["Evec"]

    T_finer = (T-0.2):0.005:(T+0.3)
    e_finer = zeros(length(T_finer))
    e2_finer = zeros(length(T_finer))

    for i in eachindex(T_finer)
        oben_e = sum(exp.(-(1/T_finer[i]-1/T) .* Evec) .* Evec)
        oben_e2 = sum(exp.(-(1/T_finer[i]-1/T) .* Evec) .* (Evec .^2))
        unten = sum(exp.(-(1/T_finer[i]-1/T) .* Evec))
        e_finer[i] = oben_e / unten
        e2_finer[i] = oben_e2 / unten
    end

    cv_finer = (e2_finer - e_finer.^2) ./ (T^2) ./ N^2
    plot(T_finer,cv_finer,label = lab)
    xlabel("Temperatur T")
    ylabel("Wärmekapazität Cv pro Spin")
    return T_finer[indmax(cv_finer)]
end

function plot_rew_demo()
    modus = 2
    samples = 10000
    N = 16
    T = 2.2
    lab = "T = $T"
    plot_rew_hat(N,T,samples,modus,lab)

    modus = 2
    samples = 10000
    N = 16
    T = 2.3
    lab = "T = $T"
    plot_rew_hat(N,T,samples,modus,lab)

    T_array = 1.0:0.02:4.0
    sweeps = [samples]
    sizes = [N]
    modus = 2
    plot_scale_cv(T_array,sizes,sweeps,modus)
    xlabel("Temperatur T")
    ylabel("Wärmekapazität Cv pro Spin")
    grid()
end

function finite_size_scaling(modus,samples,sizes,T_array)
    subplot(1,2,1)
    max_values = zeros(length(sizes))
    for i in eachindex(sizes)
        size = sizes[i]
        lab = "$size x $size"
        max_values[i] = plot_rew_hat(sizes[i],T_array[i],samples,modus,lab)
    end
    legend()
    subplot(1,2,2)
    rec_lattice = 1 ./ sizes
    rec_temperature = 1 ./ max_values
    scatter(rec_lattice,rec_temperature)

    model(x, p) = p[1] - (p[3]) .* x .^ (p[2])
    p0 = [0.44,0.5,1]
    fit = curve_fit(model, rec_lattice, rec_temperature, p0)
    print(fit.param)
    print(fit)
    beta = round(fit.param[1],3)
    v = round(fit.param[2],3)
    rec_lattice_fit = 0:0.01:0.2
    rec_temperature_fit = model(rec_lattice_fit,fit.param)
    plot(rec_lattice_fit,rec_temperature_fit,label = "beta Onsager = $beta und v = $v")
    xlabel("1/L")
    ylabel("beta max")

    legend()
    tight_layout()
end

#seed the random number generator
srand()

###plot integrated autocorrelation for metro and wolff
#T_array = 1.0:0.05:4.0;samples = 1e6;
#modus = 1
#plot_integrated_auto(T_array,samples,modus)
#modus = 2
#plot_integrated_auto(T_array,samples,modus)
#show()

###plot autocorrelation of wolff-algorithm
#samples = 1e6;
#modus = 2;
#plot_auto(samples,modus)
#xlim([0,10])
#show()

###plot autocorrelation of metropolis-algorithm
#samples = 1e6;
#modus = 1;
#plot_auto(samples,modus)
#show()

###heat_capacity with wolff algorithm
#T_array = 2.0:0.01:3.0
#modus = 2
#sweeps = [10000,10000,10000,10000,10000]
#sizes = [4,8,16,32,64]
#plot_scale_cv(T_array,sizes,sweeps,modus)
#show()

###mean energy with wolff algorithm
#T_array = 1.0:0.05:4.0
#modus = 2
#sweeps = [10000,10000,10000,10000,10000]
#sizes = [4,8,16,32,64]
#plot_scale_e(T_array,sizes,sweeps,modus)
#show()

###thermalization with wolff algorithm
#T = 1.8
#samples = 1000
#modus = 2
#Ns = [4,8,16,32,64,128,256]
#thermalization(T,Ns,samples,modus)
#xlim([0,1000])
#show()

###thermalization with metropolis algorithm
#T = 1
#samples = 50000
#modus = 1
#Ns = [4,8,16,32]
#thermalization(T,Ns,samples,modus)
#show()

###time series
#T = 1
#plot_time_series_images_metropolis(T)
#T = 1
#plot_time_series_images_wolff(T)

###heat_capacity with wolff algorithm_reweighting
#modus = 2
#samples = 20000
#sizes = [8,16,32,64]
#T_array = [2.45,2.34,2.30,2.28]
#finite_size_scaling(modus,samples,sizes,T_array)
#show()

#plot_rew_demo()
#show()
