# Canonical family definitions for the replicated multiplicity sweeps.
#
# This file is deliberately shared by the stochastic sweep and the deterministic
# cross-family analysis. Do not duplicate these split functions in new pipelines.

struct CanonicalFamilySpec
    family::String
    slug::String
    source::String
    pfam_id::String
    marker::String
    fit_included::Bool
end

const CANONICAL_FAMILIES = [
    CanonicalFamilySpec("Kunitz", "kunitz", "Pfam", "PF00014", "P1 K/R", true),
    CanonicalFamilySpec("SH3", "sh3", "Pfam", "PF00018", "Trp", true),
    CanonicalFamilySpec("WW", "ww", "Pfam", "PF00397", "Spec. loop", true),
    CanonicalFamilySpec("Homeobox", "homeobox", "Pfam", "PF00046", "Gln at pos. 50", true),
    CanonicalFamilySpec("Forkhead", "forkhead", "Pfam", "PF00250", "H/N at H3", true),
    CanonicalFamilySpec("Conotoxin", "omega_conotoxin", "SwissProt", "", "Tyr13", false),
]

function canonical_load_alignment(spec::CanonicalFamilySpec, data_dir::AbstractString)
    family_dir = joinpath(data_dir, spec.slug)
    mkpath(family_dir)
    if !isempty(spec.pfam_id)
        sto_file = download_pfam_seed(spec.pfam_id; cache_dir=family_dir)
        raw = parse_stockholm(sto_file)
        char_mat, names = clean_alignment(raw; max_gap_frac_col=0.5, max_gap_frac_seq=0.3)
        return char_mat, names, nothing
    end

    full_fasta = joinpath(family_dir, "omega_conotoxin_full_family_aligned.fasta")
    strong_fasta = joinpath(family_dir, "strong_cav22_binders_aligned.fasta")
    isfile(full_fasta) || error("Missing conotoxin alignment: $full_fasta")
    isfile(strong_fasta) || error("Missing conotoxin designated-set alignment: $strong_fasta")
    raw_full = parse_fasta(full_fasta)
    raw_strong = parse_fasta(strong_fasta)
    char_mat, names = clean_alignment(raw_full; max_gap_frac_col=0.5, max_gap_frac_seq=0.4)
    strong_ids = Set(split(name, "|")[1] for (name, _) in raw_strong)
    return char_mat, names, strong_ids
end

function canonical_split(spec::CanonicalFamilySpec, char_mat, names, auxiliary)
    family = spec.family
    K, L = size(char_mat)

    if family == "Kunitz"
        lys_fracs = [count(i -> char_mat[i, j] == 'K', 1:K) /
                     max(1, count(i -> !(char_mat[i, j] in ('-', '.')), 1:K))
                     for j in 1:L]
        marker = argmax(lys_fracs)
        group_A = findall(i -> char_mat[i, marker] in ('K', 'R'), 1:K)
        group_B = findall(i -> !(char_mat[i, marker] in ('K', 'R')) &&
                               !(char_mat[i, marker] in ('-', '.')), 1:K)
        return group_A, group_B, marker, Set(['K', 'R'])
    elseif family == "SH3"
        trp_fracs = [count(i -> char_mat[i, j] == 'W', 1:K) /
                     max(1, count(i -> !(char_mat[i, j] in ('-', '.')), 1:K))
                     for j in 1:L]
        variable_W = findall(f -> 0.15 < f < 0.85, trp_fracs)
        marker = isempty(variable_W) ? argmin(abs.(trp_fracs .- 0.5)) :
                 variable_W[argmax(trp_fracs[variable_W])]
        group_A = findall(i -> char_mat[i, marker] == 'W', 1:K)
        group_B = findall(i -> char_mat[i, marker] != 'W' &&
                               !(char_mat[i, marker] in ('-', '.')), 1:K)
        return group_A, group_B, marker, Set(['W'])
    elseif family == "WW"
        mid_start, mid_end = max(1, L ÷ 3), min(L, 2L ÷ 3)
        best_pos, best_entropy = mid_start, -Inf
        for j in mid_start:mid_end
            residues = [char_mat[i, j] for i in 1:K if !(char_mat[i, j] in ('-', '.'))]
            length(residues) < K ÷ 2 && continue
            counts = StatsBase.countmap(residues)
            total = length(residues)
            entropy = -sum((n / total) * log(n / total) for n in values(counts))
            if entropy > best_entropy
                best_pos, best_entropy = j, entropy
            end
        end
        counts = StatsBase.countmap([char_mat[i, best_pos] for i in 1:K
                                     if !(char_mat[i, best_pos] in ('-', '.'))])
        top_aa = first(sort(collect(counts), by=x -> (-x[2], x[1])))[1]
        group_A = findall(i -> char_mat[i, best_pos] == top_aa, 1:K)
        group_B = findall(i -> char_mat[i, best_pos] != top_aa &&
                               !(char_mat[i, best_pos] in ('-', '.')), 1:K)
        return group_A, group_B, best_pos, Set([top_aa])
    elseif family == "Homeobox"
        start = max(1, 2L ÷ 3)
        gln_fracs = zeros(L)
        for j in start:L
            valid = count(i -> !(char_mat[i, j] in ('-', '.')), 1:K)
            gln_fracs[j] = valid == 0 ? 0.0 :
                           count(i -> char_mat[i, j] == 'Q', 1:K) / valid
        end
        marker = argmax(gln_fracs)
        group_A = findall(i -> char_mat[i, marker] == 'Q', 1:K)
        group_B = findall(i -> char_mat[i, marker] != 'Q' &&
                               !(char_mat[i, marker] in ('-', '.')), 1:K)
        return group_A, group_B, marker, Set(['Q'])
    elseif family == "Forkhead"
        mid_start, mid_end = max(1, L ÷ 3), min(L, 2L ÷ 3)
        hn_fracs = zeros(L)
        for j in mid_start:mid_end
            valid = count(i -> !(char_mat[i, j] in ('-', '.')), 1:K)
            hn_fracs[j] = valid == 0 ? 0.0 :
                          count(i -> char_mat[i, j] in ('H', 'N'), 1:K) / valid
        end
        candidates = [j for j in mid_start:mid_end if 0.3 < hn_fracs[j] < 0.7]
        marker = isempty(candidates) ? argmax(hn_fracs) :
                 candidates[argmin(abs.(hn_fracs[candidates] .- 0.5))]
        group_A = findall(i -> char_mat[i, marker] in ('H', 'N'), 1:K)
        group_B = findall(i -> !(char_mat[i, marker] in ('H', 'N')) &&
                               !(char_mat[i, marker] in ('-', '.')), 1:K)
        return group_A, group_B, marker, Set(['H', 'N'])
    elseif family == "Conotoxin"
        strong_ids = auxiliary
        group_A = findall(i -> split(names[i], "|")[1] in strong_ids, 1:K)
        group_B = setdiff(collect(1:K), group_A)
        tyr_fracs = [begin
            valid = count(i -> !(char_mat[i, j] in ('-', '.', '~')), 1:K)
            valid == 0 ? 0.0 : count(i -> char_mat[i, j] == 'Y', 1:K) / valid
        end for j in 1:L]
        marker = argmax(tyr_fracs)
        return group_A, group_B, marker, Set(['Y'])
    end
    error("No canonical split for $(spec.family)")
end

function canonical_separation(X, group_A, group_B)
    similarities(indices_1, indices_2; same_group=false) = [
        dot(X[:, i], X[:, j]) / (norm(X[:, i]) * norm(X[:, j]))
        for i in indices_1 for j in indices_2 if !same_group || i < j
    ]
    within_A = similarities(group_A, group_A; same_group=true)
    within_B = similarities(group_B, group_B; same_group=true)
    between = similarities(group_A, group_B)
    total_within = length(within_A) + length(within_B)
    mean_within = (mean(within_A) * length(within_A) +
                   mean(within_B) * length(within_B)) / total_within
    std_within = sqrt((var(within_A) * length(within_A) +
                       var(within_B) * length(within_B)) / total_within)
    return (mean_within - mean(between)) / (0.5 * (std_within + std(between)))
end
