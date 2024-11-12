def computeCURBD(sim):
    """
    function [CURBD,CURBDLabels] = computeCURBD(varargin)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % Performs Current-Based Decomposition (CURBD) of multi-region data. Ref:
    %
    % Perich MG et al. Inferring brain-wide interactions using data-constrained
    % recurrent neural network models. bioRxiv. DOI:
    %
    % Two input options:
    %   1) out = computeCURBD(model, params)
    %       Pass in the output struct of trainMultiRegionRNN and it will do the
    %       current decomposition. Note that regions has to be defined.
    %
    %   2) out = computeCURBD(RNN, J, regions, params)
    %       Only needs the RNN activity, region info, and J matrix
    %
    %   Only parameter right now is current_type, to isolate excitatory or
    %   inhibitory currents.
    %
    % OUTPUTS:
    %   CURBD: M x M cell array containing the decomposition for M regions.
    %       Target regions are in rows and source regions are in columns.
    %   CURBDLabels: M x M cell array with string labels for each current
    %
    %
    % Written by Matthew G. Perich. Updated December 2020.
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    current_type = 'all'  # 'excitatory', 'inhibitory', or 'all'
    RNN = sim['RNN']
    J = sim['J'].copy()
    regions = sim['regions']

    if regions is None:
        raise ValueError("regions not specified")

    if current_type == 'excitatory':  # take only positive J weights
        J[J < 0] = 0
    elif current_type == 'inhibitory':  # take only negative J weights
        J[J > 0] = 0
    elif current_type == 'all':
        pass
    else:
        raise ValueError("Unknown current type: {}".format(current_type))

    nRegions = regions.shape[0]

    # loop along all bidirectional pairs of regions
    CURBD = {(i, j): None for i in range(nRegions) for j in range(nRegions)}
    #CURBD = np.empty((nRegions, nRegions), dtype=np.object)
    #CURBDLabels = np.empty((nRegions, nRegions), dtype=np.object)
    CURBDLabels = {(i, j): None for i in range(nRegions) for j in range(nRegions)}

    for key in CURBD.keys():
        idx_trg, idx_src = key
        in_trg = regions[idx_trg, 1]
        lab_trg = regions[idx_trg, 0]
        in_src = regions[idx_src, 1]
        lab_src = regions[idx_src, 0]
        sub_J = J[in_trg, :][:, in_src]
        CURBD[(idx_trg, idx_src)] = sub_J.dot(RNN[in_src, :])
        CURBDLabels[(idx_trg, idx_src)] = "{} to {}".format(lab_src,
                                                            lab_trg)

    return (CURBD, CURBDLabels)
