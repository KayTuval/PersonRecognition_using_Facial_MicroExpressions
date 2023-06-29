__author__ = 'Khen Cohen'
__credits__ = ['Khen Cohen']
__email__ = 'khencohen@mail.tau.ac.il'
__date__ = '5.10.2021'

from khen_hardware_libs.SignalClass.SignalClass import *

def get_Smat(b_vec, g_vec, r_vec):
    return np.concatenate((b_vec.reshape(-1, 1), \
                           g_vec.reshape(-1, 1), \
                           r_vec.reshape(-1, 1)), axis=1)

def get_invMmat(N):
    m_mat = np.zeros((N, N))
    for i in range(N):
        m_mat[i, i] = 2
        if i > 0:
            m_mat[i - 1, i] = -1
        if i < N - 1:
            m_mat[i + 1, i] = -1

    return np.linalg.inv(2 * m_mat)

def get_Itrans(b_vec, g_vec, r_vec, N):
    Smat = get_Smat(b_vec, g_vec, r_vec)
    invMmat = get_invMmat(N)
    return invMmat @ Smat @ np.linalg.pinv(Smat.transpose() @ invMmat @ Smat)


def integrate_interval(f_fun, t1, t2):
    return f_fun( np.linspace(t1,t2, 100 ) ).mean()#*(t2-t1)

def sample_function(f_fun, time_vec):
    signal_vec = []
    for i in range(len(time_vec)-1):
        signal_vec += [ integrate_interval(f_fun, time_vec[i], time_vec[i+1]) ]
    return signal_vec

def simulate_rgb_sample(f_fun, time_vec, b_vec, g_vec, r_vec):
    signal_vec = []

    N = b_vec.shape[0]
    Itrans = get_Itrans(b_vec, g_vec, r_vec, N)

    for i in range(len(time_vec)-1):
        delta = ( time_vec[i+1] - time_vec[i] ) / N
        B = 0;  G = 0;    R = 0
        for j in range(N):
            B += b_vec[j]*integrate_interval(f_fun, time_vec[i]+delta*j, time_vec[i]+delta*(j+1))
            G += g_vec[j]*integrate_interval(f_fun, time_vec[i]+delta*j, time_vec[i]+delta*(j+1))
            R += r_vec[j]*integrate_interval(f_fun, time_vec[i]+delta*j, time_vec[i]+delta*(j+1))
        # B /= b_vec.sum()    # TODO: check normalization
        # G /= g_vec.sum()
        # R /= r_vec.sum()
        # for j in range(N):
        #     signal_vec += [ integrate_interval(f_fun, time_vec[i] + delta*j, time_vec[i]+(j+1)*delta) ]

        # print(B,G,R)
        BGR = Itrans @ np.array([B,G,R]).reshape(-1,1)
        # print(BGR)
        # BGR = np.array([B/2, G/2, R/2, (B+G+R)/2]).reshape(-1, 1)

        signal_vec += list(BGR.reshape(-1))

    return signal_vec

def get_bgr_vectors_from_N(N):
    if N == 3:
        b_vec = np.array([1, 0, 0])
        g_vec = np.array([0, 1, 0])
        r_vec = np.array([0, 0, 1])
    elif N == 4:
        b_vec = np.array([1, 0, 0, 1])
        g_vec = np.array([1, 0, 1, 0])
        r_vec = np.array([0, 1, 0, 1])
    elif N == 5:
        b_vec = np.array([0, 1, 0, 0, 0])
        g_vec = np.array([0, 0, 0, 1, 0])
        r_vec = np.array([1, 0, 1, 0, 1])
    elif N == 6:
        b_vec = np.array([1, 0, 1, 0, 1, 0])
        g_vec = np.array([0, 1, 0, 1, 0, 1])
        r_vec = np.array([1, 1, 1, 1, 1, 1])
    return b_vec, g_vec, r_vec



def compare_x1_x3_x4_x5_x6_score(epochs, min_f = 0, max_f = 0, super_time_step = 0):
    total_freq_vec = []
    total_l2_score_vec = []
    # total_pearson_score_vec = []

    for epoch in range(epochs):
        freq = min_f + np.random.rand() * (max_f-min_f)
        f_fun = lambda t: freq * np.sin(freq * 2 * np.pi * t)

        print('epoch =', epoch)
        #### True Signal #####
        sigTrue = SignalClass(np.array(f_fun(super_time_vec[:-int(1 / super_time_step)])),
                              time_step=super_time_step / camera_fps, name='True Signal')

        #### N = 1 #####
        signal_vec_x1 = sample_function(f_fun, time_vec)

        #### N = 3 #####
        b_vec, g_vec, r_vec = get_bgr_vectors_from_N(3)
        signal_vec_x3 = simulate_rgb_sample(f_fun, time_vec, b_vec, g_vec, r_vec)

        #### N = 4 #####
        # b_vec = np.array([0, 1, 0, 0])
        # g_vec = np.array([1, 0, 0, 1])
        # r_vec = np.array([0, 0, 1, 0])
        b_vec, g_vec, r_vec = get_bgr_vectors_from_N(4)
        signal_vec_x4 = simulate_rgb_sample(f_fun, time_vec, b_vec, g_vec, r_vec)

        #### N = 5 #####
        b_vec, g_vec, r_vec = get_bgr_vectors_from_N(5)
        signal_vec_x5 = simulate_rgb_sample(f_fun, time_vec, b_vec, g_vec, r_vec)

        #### N = 6 #####
        # b_vec = np.array([1, 0, 0, 0, 0, 1])
        # g_vec = np.array([0, 1, 1, 0, 0, 0])
        # r_vec = np.array([0, 0, 0, 1, 1, 0])
        # b_vec = np.array([0, 1, 0, 0, 0, 0])
        # g_vec = np.array([1, 0, 1, 1, 0, 1])
        # r_vec = np.array([0, 0, 0, 0, 1, 0])
        b_vec, g_vec, r_vec = get_bgr_vectors_from_N(6)
        signal_vec_x6 = simulate_rgb_sample(f_fun, time_vec, b_vec, g_vec, r_vec)

        # # Before temporal up-sampling:
        # plt.plot(super_time_vec, f_fun(super_time_vec), color = 'b')
        # plt.plot(np.arange(0,T,1./camera_fps)[:-1] + 0.5/camera_fps, signal_vec_x1)#, color = 'g')
        # plt.plot(np.arange(0,T,1./(3*camera_fps))[:-3] + 0.5/(3*camera_fps), signal_vec_x3)#, color = 'r')
        # plt.plot(np.arange(0,T,1./(4*camera_fps))[:-4] + 0.5/(4*camera_fps), signal_vec_x4)
        # plt.plot(np.arange(0,T,1./(5*camera_fps))[:-5] + 0.5/(5*camera_fps), signal_vec_x5)
        # plt.plot(np.arange(0,T,1./(6*camera_fps))[:-6] + 0.5/(6*camera_fps), signal_vec_x6)
        # plt.legend(['True signal', 'x1', 'x3', 'x4', 'x5', 'x6'])
        # plt.show()
        # exit()

        # Temporal up-sampling:
        signal_vec_x1 = temporal_dense_vec(signal_vec_x1, int(1 / (1 * super_time_step)))
        signal_vec_x3 = temporal_dense_vec(signal_vec_x3, int(1 / (3 * super_time_step)))
        signal_vec_x4 = temporal_dense_vec(signal_vec_x4, int(1 / (4 * super_time_step)))
        signal_vec_x5 = temporal_dense_vec(signal_vec_x5, int(1 / (5 * super_time_step)))
        signal_vec_x6 = temporal_dense_vec(signal_vec_x6, int(1 / (6 * super_time_step)))

        sig1x = signalClass(np.array(signal_vec_x1), time_step=super_time_step / camera_fps, name='x1')
        sig3x = signalClass(np.array(signal_vec_x3), time_step=super_time_step / camera_fps, name='x3')
        sig4x = signalClass(np.array(signal_vec_x4), time_step=super_time_step / camera_fps, name='x4')
        sig5x = signalClass(np.array(signal_vec_x5), time_step=super_time_step / camera_fps, name='x5')
        sig6x = signalClass(np.array(signal_vec_x6), time_step=super_time_step / camera_fps, name='x6')

        # plt.plot(super_time_vec, f_fun(super_time_vec), color = 'b')
        # plt.plot(super_time_vec[:-int(1/(1*super_time_step))] + 0.5/camera_fps, signal_vec_x1)#, color = 'g')
        # plt.plot(super_time_vec[:-3*int(1/(3*super_time_step))] + 0.5/(3*camera_fps), signal_vec_x3)#, color = 'r')
        # plt.plot(super_time_vec[:-4*int(1/(4*super_time_step))] + 0.5/(4*camera_fps), signal_vec_x4)
        # plt.plot(super_time_vec[:-5*int(1/(5*super_time_step))] + 0.5/(5*camera_fps), signal_vec_x5)
        # plt.plot(super_time_vec[:-6*int(1/(6*super_time_step))] + 0.5/(6*camera_fps), signal_vec_x6)
        # plt.legend(['True signal', 'x1', 'x3', 'x4', 'x5', 'x6'])
        # plt.xlim([1.0,1.5])
        # plt.show()

        # plt.plot(sigTrue.signal[int(0.5/(3*super_time_step)):] )
        # # plt.plot(sig1x.signal[:-int(0.5/(super_time_step))] )
        # plt.plot(sig3x.signal[:-int(0.5/(3*super_time_step))] )
        # plt.show()

        # Compare perforemence:
        l2_scores = [sigTrue.l2_distance(sig1x, int(0.5 / (super_time_step))), \
                     sigTrue.l2_distance(sig3x, int(0.5 / (3 * super_time_step))), \
                     sigTrue.l2_distance(sig4x, int(0.5 / (4 * super_time_step))), \
                     sigTrue.l2_distance(sig5x, int(0.5 / (5 * super_time_step))), \
                     sigTrue.l2_distance(sig6x, int(0.5 / (6 * super_time_step)))]

        # print(l2_scores)
        # pearson_scores = [sigTrue.pearson_correlation(sig1x)[0], sigTrue.pearson_correlation(sig3x)[0],
        #                   sigTrue.pearson_correlation(sig4x)[0], sigTrue.pearson_correlation(sig5x)[0], \
        #                   sigTrue.pearson_correlation(sig6x)[0]]

        total_freq_vec += [[freq] * 5]
        total_l2_score_vec += [l2_scores]
        # total_pearson_score_vec += [pearson_scores]

    plt.plot(np.array(total_freq_vec), np.array(total_l2_score_vec), 'o')
    plt.legend(['x1', 'x3', 'x4', 'x5', 'x6'])
    plt.title('L2 Score')
    plt.savefig('Plots/x1_x3_x4_x5_x6_comparison.png')
    plt.show()

    return


def compare_fixed_pattern_flicker_vec(epochs, min_f, max_f, N = 4, super_time_step = 0):
    if N == 4:
        flicker_mat_vec = [ np.array([[1, 0, 0, 1],\
                                      [0, 1, 0, 0],\
                                      [0, 0, 1, 0]]),\
                            np.array([[1, 0, 0, 1], \
                                      [1, 0, 1, 0], \
                                      [0, 1, 0, 1]]),\
                            np.array([[1, 0, 0, 0], \
                                      [0, 1, 1, 0], \
                                      [0, 0, 0, 1]]),\
                            np.array([[1, 1, 0, 0], \
                                      [0, 1, 1, 0], \
                                      [0, 0, 1, 1]]),\
                            np.array([[0, 0, 1, 0], \
                                      [0, 1, 0, 0], \
                                      [1, 1, 1, 1]]) ]
    elif N == 5:
        flicker_mat_vec = [ np.array([[1, 0, 0, 0, 1],\
                                      [0, 1, 1, 0, 0],\
                                      [0, 0, 1, 1, 0]]),\
                            np.array([[1, 0, 0, 1, 0], \
                                      [1, 0, 1, 0, 1], \
                                      [0, 1, 0, 0, 1]]),\
                            np.array([[1, 0, 0, 1, 0], \
                                      [0, 0, 1, 0, 0], \
                                      [0, 1, 0, 0, 1]]),\
                            np.array([[0, 1, 0, 0, 0], \
                                      [1, 0, 1, 0, 1], \
                                      [0, 0, 0, 1, 0]]),\
                            np.array([[1, 1, 0, 0, 0], \
                                      [0, 1, 1, 1, 0], \
                                      [0, 0, 0, 1, 1]]) ]

    elif N == 6:
        flicker_mat_vec = [ np.array([[1, 0, 0, 0, 1, 0],\
                                      [0, 1, 0, 0, 0, 1],\
                                      [0, 0, 1, 1, 0, 0]]),\
                            np.array([[1, 1, 0, 0, 0, 0], \
                                      [0, 0, 1, 1, 0, 0], \
                                      [0, 0, 0, 0, 1, 1]]),\
                            np.array([[1, 0, 0, 1, 0, 0], \
                                      [0, 1, 1, 1, 1, 0], \
                                      [0, 0, 1, 0, 0, 1]]),\
                            np.array([[1, 0, 1, 0, 1, 0], \
                                      [0, 1, 0, 1, 0, 1], \
                                      [1, 1, 1, 1, 1, 1]]),\
                            np.array([[0, 1, 0, 0, 0, 0], \
                                      [1, 0, 1, 1, 0, 1], \
                                      [0, 0, 0, 0, 1, 0]]) ]



    freq_dict = {}
    score_dict = {}
    for i in range(len(flicker_mat_vec)):
        freq_dict[i] = []
        score_dict[i] = []

    for epoch in range(epochs):
        freq = min_f + np.random.rand() * (max_f-min_f)
        f_fun = lambda t: freq * np.sin(freq * 2 * np.pi * t)
        print('epoch =', epoch)
        #### True Signal #####
        sigTrue = SignalClass(np.array(f_fun(super_time_vec[:-int(1 / super_time_step)])),
                              time_step=super_time_step / camera_fps, name='True Signal')

        #### N #####
        b_vec, g_vec, r_vec = flicker_mat_vec[ epoch % len(flicker_mat_vec) ]
        signal_vec_x = simulate_rgb_sample(f_fun, time_vec, b_vec, g_vec, r_vec)

        # Temporal up-sampling:
        signal_vec_x = temporal_dense_vec(signal_vec_x, int(1 / (N * super_time_step)))
        sig4x = SignalClass(np.array(signal_vec_x), time_step=super_time_step / camera_fps, name='x'+str(N))

        # Compare perforemence:
        l2_scores = sigTrue.l2_distance(sig4x, int(0.5 / (N * super_time_step)))

        freq_dict[ epoch % len(flicker_mat_vec) ] += [freq]
        score_dict[ epoch % len(flicker_mat_vec) ] += [l2_scores]

    for i in range(len(flicker_mat_vec)):
        plt.plot(np.array(freq_dict[i]), np.array(score_dict[i]), 'o')
    plt.legend(range(len(flicker_mat_vec)))
    plt.title('L2 Score')
    plt.savefig('Plots/x'+str(N)+'_fixed_pattern_flicker.png')
    plt.show()

    return

def compare_x1_x3_x4_flicker_random_vec(epochs, min_f, max_f):
    def generate_flicker_vec_randomly(N):
        flag = True
        while flag:
            mat = np.int8(np.random.rand(3,N)+0.5)
            if np.linalg.matrix_rank(mat) == 3:
                flag = False
        b_vec = mat[0,:]
        g_vec = mat[0,:]
        r_vec = mat[0,:]
        # print(mat)
        return (b_vec, g_vec, r_vec)

    total_freq_vec = []
    total_l2_score_vec = []
    for epoch in range(epochs):
        freq = min_f + np.random.rand() * (max_f-min_f)
        f_fun = lambda t: freq * np.sin(freq * 2 * np.pi * t)
        print('epoch =', epoch)
        #### True Signal #####
        sigTrue = SignalClass(np.array(f_fun(super_time_vec[:-int(1 / super_time_step)])),
                              time_step=super_time_step / camera_fps, name='True Signal')
        #### N = 1 #####
        signal_vec_x1 = sample_function(f_fun, time_vec)
        #### N = 3 #####
        b_vec, g_vec, r_vec = get_bgr_vectors_from_N(3)
        signal_vec_x3 = simulate_rgb_sample(f_fun, time_vec, b_vec, g_vec, r_vec)
        #### N = 4 #####
        b_vec, g_vec, r_vec = generate_flicker_vec_randomly(4)
        signal_vec_x4 = simulate_rgb_sample(f_fun, time_vec, b_vec, g_vec, r_vec)

        # Temporal up-sampling:
        signal_vec_x1 = temporal_dense_vec(signal_vec_x1, int(1 / (1 * super_time_step)))
        signal_vec_x3 = temporal_dense_vec(signal_vec_x3, int(1 / (3 * super_time_step)))
        signal_vec_x4 = temporal_dense_vec(signal_vec_x4, int(1 / (4 * super_time_step)))
        sig1x = SignalClass(np.array(signal_vec_x1), time_step=super_time_step / camera_fps, name='x1')
        sig3x = SignalClass(np.array(signal_vec_x3), time_step=super_time_step / camera_fps, name='x3')
        sig4x = SignalClass(np.array(signal_vec_x4), time_step=super_time_step / camera_fps, name='x4')

        # Compare perforemence:
        l2_scores = [sigTrue.l2_distance(sig1x, int(0.5 / (super_time_step))), \
                     sigTrue.l2_distance(sig3x, int(0.5 / (3 * super_time_step))), \
                     sigTrue.l2_distance(sig4x, int(0.5 / (4 * super_time_step)))]

        total_freq_vec += [[freq] * 3]
        total_l2_score_vec += [l2_scores]

    plt.plot(np.array(total_freq_vec), np.array(total_l2_score_vec), 'o')
    plt.legend(['x1', 'x3', 'x4 - random flicker'])
    plt.title('L2 Score')
    plt.savefig('Plots/x4_random_flicker.png')
    plt.show()

    return


def evaluate_xN_score(N, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, white_noise_filter = 0.0, folder_name =''):
    # This function gets N up sampling factor and fun function of the signal. It returns the up-sampling method result.

    #### N #####
    b_vec, g_vec, r_vec = get_bgr_vectors_from_N(N)

    true_signal_vec = np.array(f_fun(super_time_vec))
    signal_vec_x1 = sample_function(f_fun, time_vec)
    signal_vec_xN = simulate_rgb_sample(f_fun, time_vec, b_vec, g_vec, r_vec)

    #### N = 1 #####
    signal_vec_x1 = temporal_dense_vec(signal_vec_x1, int(1 / (1 * super_time_step)))

    ##### N ####
    signal_vec_xN = temporal_dense_vec(signal_vec_xN, int(1 / (N * super_time_step)))

    #### True Signal #####
    sigTrue = SignalClass(true_signal_vec, time_step=super_time_step / camera_fps, name='True Signal')
    sig1x = SignalClass(np.array(signal_vec_x1), time_step=super_time_step / camera_fps, name='x1')
    sigNx = SignalClass(np.array(signal_vec_xN), time_step=super_time_step / camera_fps, name='x'+str(N))

    sig1x.time_vec = sig1x.time_vec[int(0.5 / (1 * super_time_step)):]
    sig1x.signal = sig1x.signal[:-int(0.5 / (1 * super_time_step))]
    sigNx.time_vec = sigNx.time_vec[int(0.5 / (N * super_time_step)):]
    sigNx.signal = sigNx.signal[:-int(0.5 / (N * super_time_step))]
    sig1x.set_signal(sig1x.signal)
    sigNx.set_signal(sigNx.signal)

    sig1x.filter_white_noise(white_noise_filter)
    sigNx.filter_white_noise(white_noise_filter)

    fig = plt.figure(figsize=(15, 5))
    # plt.subplot(1,2,1)
    # plt.plot(sigTrue.time_vec, np.real(sigTrue.signal), '-')
    # plt.plot(sig1x.time_vec, np.real(sig1x.signal), '-')
    # plt.plot(sigNx.time_vec, np.real(sigNx.signal), 'r-')
    # plt.legend(['True signal', 'x1', 'x'+str(N)])
    # plt.title('Temporal Comparison - Simulation')
    # # plt.savefig('Plots/simulation_temporal_N='+str(N)+'.png')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Signal')
    # plt.xlim([2.0,4.0])
    #
    # plt.subplot(1,2,2)
    plt.plot(sigTrue.frequencies, np.abs(sigTrue.signal_fft), '-')
    plt.plot(sig1x.frequencies, np.abs(sig1x.signal_fft), '-')
    plt.plot(sigNx.frequencies, np.abs(sigNx.signal_fft), 'r-')
    plt.legend(['True signal', 'x1', 'x'+str(N)])
    plt.title('Spectrum Comparison - Simulation')
    # plt.savefig('Plots/simulation_spectrum_N='+str(N)+'.png')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Signal')
    plt.xlim([-30.0,30.0])
    plt.savefig('Plots/'+folder_name+'/simulation_temporal_spectrum_N='+str(N)+'.png')
    # plt.show()
    plt.close()
    plt.clf()

    return

def perform_scanning_method(temporal_window, N_list, f_fun, super_time_step, camera_fps, time_vec, \
                            super_time_vec, white_noise_filter, apply_anti_aliasing, folder_name):

    true_signal_vec = np.array(f_fun(super_time_vec))
    sigTrue = SignalClass(true_signal_vec, time_step=super_time_step / camera_fps, name='True Signal')
    total_signal_vec = []
    total_time_vec = []
    for i in range(len(N_list)):
        N = N_list[i]
        b_vec, g_vec, r_vec = get_bgr_vectors_from_N(N)
        start_time = i*temporal_window
        stop_time = (i+1)*temporal_window
        cur_time_vec = time_vec[ np.abs( time_vec - start_time ).argmin() : np.abs( time_vec - stop_time ).argmin() ]

        signal_vec_xN = simulate_rgb_sample(f_fun, cur_time_vec, b_vec, g_vec, r_vec)
        signal_vec_xN = temporal_dense_vec(signal_vec_xN, int(1 / (N * super_time_step)))
        # signal_vec_xN = temporal_dense_vec(signal_vec_xN, int(T/temporal_window) )
        sigNx = SignalClass(np.array(signal_vec_xN), time_step=super_time_step / camera_fps, name='x' + str(N))

        # ----- Extract the relevant part from the signal -----
        filter_signal = SignalClass(sigNx.time_vec, time_step=super_time_step / camera_fps, name='Filter')
        # if N == 1:
        #     filter_vec = get_LPF(filter_signal.frequencies, cutoff_freq=0.5 * camera_fps)
        if N == 3:
            filter_vec = get_LPF(filter_signal.frequencies, cutoff_freq=0.5 * camera_fps * 3)
        elif N == 4:
            filter_vec = get_BPF(filter_signal.frequencies, 0.5 * camera_fps * 3, 0.5 * camera_fps * 4)
        elif N == 5:
            filter_vec = get_BPF(filter_signal.frequencies, 0.5 * camera_fps * 4, 0.5 * camera_fps * 5)
        elif N == 6:
            filter_vec = get_BPF(filter_signal.frequencies, 0.5 * camera_fps * 5, 0.5 * camera_fps * 6)
        filter_signal.set_signal_by_spectrum( filter_vec )
        intensity_before = np.abs(sigNx.signal).sum()
        sigNx = sigNx * filter_signal
        # sigNx.show_spectrum()
        intensity_after = np.abs(sigNx.signal).sum()
        sigNx = sigNx * (intensity_before/intensity_after)  # For spectrum filtering attenuation
        # sigNx = sigNx * N                                   # For each temporal slice
        # print( N*intensity_before/intensity_after )
        # sigNx = sigNx * (1/N_list.count(N)) * N
        # -----------------------------------------------------

        sigNx.time_vec = sigNx.time_vec[int(0.5 / (N * super_time_step)):]
        sigNx.signal = sigNx.signal[:-int(0.5 / (N * super_time_step))]
        sigNx.set_signal(sigNx.signal)
        total_time_vec += [start_time+sigNx.time_vec.copy()]
        total_signal_vec += [sigNx.signal.copy()]

    total_time_vec = np.concatenate(total_time_vec)
    total_signal_vec = np.concatenate(total_signal_vec)
    # total_signal_spectrum_vec = np.concatenate(total_signal_spectrum_vec)
    total_signal = SignalClass(total_signal_vec, time_step=super_time_step / camera_fps, name='Scaning Recovered Signal')
    # total_signal.set_signal_by_spectrum(total_signal_spectrum_vec)
    total_signal.time_vec = total_time_vec

    # ---- Anti-Aliasing Refinement: ------
    if apply_anti_aliasing:
        total_signal.set_signal( np.real(total_signal.signal) )
        total_signal.clean_phase()
        # From the highest frequency to the lowest:
        # sorted_N_list = N_list.copy()
        # sorted_N_list.sort().reverse()
        # Assumes N includes 3,4,5,6
        # N = 5:
        if max(N_list) >= 6:
            # total_signal.show_spectrum(range=[-31,31])
            filter_signal.set_signal_by_spectrum( get_BPF(total_signal.frequencies, 0.5 * camera_fps * 5, 0.5 * camera_fps * 6) )
            filtered_signal = total_signal * filter_signal
            filtered_signal = filtered_signal.rotate_spectrum(0.5 * camera_fps * 5, 0.5 * camera_fps)
            # filtered_signal.show_spectrum(range=[-31,31])
            filtered_signal.clean_phase()
            # total_signal.show_spectrum(range=[-31, 31])
            total_signal = total_signal - filtered_signal
            # total_signal.show_spectrum(range=[-31, 31])
        # N = 4:
        if max(N_list) >= 5:
            filter_signal.set_signal_by_spectrum( get_BPF(total_signal.frequencies, 0.5 * camera_fps * 4, 0.5 * camera_fps * 5) )
            filtered_signal = total_signal * filter_signal
            filtered_signal = filtered_signal.rotate_spectrum(0.5 * camera_fps * 4, 0.5 * camera_fps)
            filtered_signal.clean_phase()
            total_signal = total_signal - filtered_signal
        # N = 3:
        if max(N_list) >= 4:
            filter_signal.set_signal_by_spectrum( get_BPF(total_signal.frequencies, 0.5 * camera_fps * 3, 0.5 * camera_fps * 4) )
            filtered_signal = total_signal * filter_signal
            filtered_signal = filtered_signal.rotate_spectrum(0.5 * camera_fps * 3, 0.5 * camera_fps)
            filtered_signal.clean_phase()
            total_signal = total_signal - filtered_signal
        # total_signal.show_spectrum(range=[-31,31])
        # exit()
        # anti_aliasing_filter.name = 'Anti_aliasing'
    # ##############
    total_signal.filter_white_noise(white_noise_filter)
    total_signal = total_signal * N


    fig = plt.figure(figsize=(15, 5))
    # plt.subplot(1,2,1)
    # plt.plot(sigTrue.time_vec, np.real(sigTrue.signal), '-', linewidth = 1.5)
    # plt.plot(total_signal.time_vec, np.real(total_signal.signal), '-', linewidth = 1)
    # plt.legend(['True signal', 'Scanning method: '+str(N_list)])
    # plt.title('Temporal Comparison - Simulation')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Signal')
    # plt.xlim([4.0,6.0])
    #
    # plt.subplot(1,2,2)
    plt.plot(sigTrue.frequencies, np.abs(sigTrue.signal_fft), 'b-', linewidth = 1.5)
    plt.plot(total_signal.frequencies, np.abs(total_signal.signal_fft), 'r-', linewidth = 1)
    plt.legend(['True signal', 'Scanning method: '+str(N_list)])
    plt.title('The Scanning method: '+str(N_list)+' Spectrum Comparison - Simulation')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Signal')
    plt.xlim([-32.0,32.0])
    plt.savefig('Plots/'+folder_name+'/Scanning_method_N='+str(N_list)+'_antialiasing='+str(apply_anti_aliasing)+'.png')
    # plt.show()
    plt.close()
    plt.clf()

    return

# T = 5
T = 10.0
camera_fps = 10
time_vec = np.arange(0,T,1./camera_fps)

#### True Signal #####
super_time_step = 1/6000
super_time_vec = np.arange(0.0,T,super_time_step/camera_fps)
shift = int((0.5/camera_fps) / (0.001/camera_fps))


#### Random S comparison ####
# compare_x1_x3_x4_flicker_random_vec(3000, 5, 30, super_time_step)

##### Fix S comparison #####
# compare_fixed_pattern_flicker_vec(10000, 5, 30, N = 4, super_time_step)
# compare_fixed_pattern_flicker_vec(10000, 5, 30, N = 5, super_time_step)
# compare_fixed_pattern_flicker_vec(10000, 5, 30, N = 6, super_time_step)



##### N methods comparison #####
# compare_x1_x3_x4_x5_x6_score(10000, 5, 30, super_time_step)



##### Scanning method check #####

# 0.
# f_fun = lambda t: np.sin(3*2*np.pi*t)
# N_list = [3,4]
# temporal_window = T/len(N_list)
# perform_scanning_method(temporal_window, N_list, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, T, '')

# 1.
f_fun = lambda t: np.sin(13*2*np.pi*t) + np.sin(19*2*np.pi*t)
folder_name = 'scanning_mode_13_19'
N_list = [3,4]
temporal_window = T/len(N_list)
white_noise_filter = 0.15
apply_anti_aliasing = False
# apply_anti_aliasing = True
for i in N_list:
    evaluate_xN_score(i, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, white_noise_filter, folder_name)
perform_scanning_method(temporal_window, N_list, f_fun, super_time_step, camera_fps, time_vec, \
                        super_time_vec, white_noise_filter, apply_anti_aliasing, folder_name)

# # 2.
# f_fun = lambda t: signal.square(11*2*np.pi*t)+signal.square(17*2*np.pi*t)+signal.square(21*2*np.pi*t)
# folder_name = 'scanning_mode_square_11_17_21'
# N_list = [3,4,5]
# temporal_window = T/len(N_list)
# white_noise_filter = 0.1
# # apply_anti_aliasing = False
# apply_anti_aliasing = True
# for i in N_list:
#     evaluate_xN_score(i, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, white_noise_filter, folder_name)
# perform_scanning_method(temporal_window, N_list, f_fun, super_time_step, camera_fps, time_vec, \
#                         super_time_vec, white_noise_filter, apply_anti_aliasing, folder_name)

# # 3.
# f_fun = lambda t: signal.square(12*2*np.pi*t)+signal.square(19*2*np.pi*t)+signal.square(23*2*np.pi*t)+signal.square(27*2*np.pi*t)
# folder_name = 'scanning_mode_square_12_19_23_27'
# N_list = [3,4,5,6]
# temporal_window = T/len(N_list)
# white_noise_filter = 0.06
# # apply_anti_aliasing = False
# apply_anti_aliasing = True
# for i in N_list:
#     evaluate_xN_score(i, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, white_noise_filter, folder_name)
# perform_scanning_method(temporal_window, N_list, f_fun, super_time_step, camera_fps, time_vec, \
#                         super_time_vec, white_noise_filter, apply_anti_aliasing, folder_name)


# ##### Different N comparison examples #####
# f_fun = lambda t: np.sin(1*2*np.pi*t) + np.sin(6*2*np.pi*t) + np.sin(11*2*np.pi*t)
# evaluate_xN_score(3, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, '1_6_11')
# evaluate_xN_score(4, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, '1_6_11')
# evaluate_xN_score(5, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, '1_6_11')
# evaluate_xN_score(6, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, '1_6_11')
# f_fun = lambda t: np.sin(7*2*np.pi*t) + np.sin(12*2*np.pi*t) + np.sin(17*2*np.pi*t)
# evaluate_xN_score(3, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, '7_12_17')
# evaluate_xN_score(4, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, '7_12_17')
# evaluate_xN_score(5, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, '7_12_17')
# evaluate_xN_score(6, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, '7_12_17')
# f_fun = lambda t: np.sin(3*2*np.pi*t) + np.sin(11*2*np.pi*t) + np.sin(23*2*np.pi*t)
# evaluate_xN_score(3, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, '3_11_23')
# evaluate_xN_score(4, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, '3_11_23')
# evaluate_xN_score(5, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, '3_11_23')
# evaluate_xN_score(6, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, '3_11_23')
# f_fun = lambda t: np.sin(7*2*np.pi*t) + np.sin(28*2*np.pi*t)
# evaluate_xN_score(3, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, '7_28')
# evaluate_xN_score(4, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, '7_28')
# evaluate_xN_score(5, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, '7_28')
# evaluate_xN_score(6, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, '7_28')
# from scipy import signal
# f_fun = lambda t: signal.square(5*2*np.pi*t)
# evaluate_xN_score(3, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_5')
# evaluate_xN_score(4, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_5')
# evaluate_xN_score(5, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_5')
# evaluate_xN_score(6, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_5')
# f_fun = lambda t: signal.square(10*2*np.pi*t)
# evaluate_xN_score(3, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_10')
# evaluate_xN_score(4, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_10')
# evaluate_xN_score(5, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_10')
# evaluate_xN_score(6, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_10')
# f_fun = lambda t: signal.square(15*2*np.pi*t)
# evaluate_xN_score(3, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_15')
# evaluate_xN_score(4, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_15')
# evaluate_xN_score(5, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_15')
# evaluate_xN_score(6, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_15')
# f_fun = lambda t: signal.square(20*2*np.pi*t)
# evaluate_xN_score(3, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_20')
# evaluate_xN_score(4, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_20')
# evaluate_xN_score(5, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_20')
# evaluate_xN_score(6, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_20')
# f_fun = lambda t: signal.square(25*2*np.pi*t)
# evaluate_xN_score(3, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_25')
# evaluate_xN_score(4, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_25')
# evaluate_xN_score(5, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_25')
# evaluate_xN_score(6, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_25')
# f_fun = lambda t: signal.square(30*2*np.pi*t)
# evaluate_xN_score(3, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_30')
# evaluate_xN_score(4, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_30')
# evaluate_xN_score(5, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_30')
# evaluate_xN_score(6, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_30')
# f_fun = lambda t: signal.square(5*2*np.pi*t) + signal.square(10*2*np.pi*t)
# evaluate_xN_score(3, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_5_10')
# evaluate_xN_score(4, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_5_10')
# evaluate_xN_score(5, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_5_10')
# evaluate_xN_score(6, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_5_10')
# f_fun = lambda t: signal.square(11*2*np.pi*t) + signal.square(27*2*np.pi*t)
# evaluate_xN_score(3, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_11_27')
# evaluate_xN_score(4, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_11_27')
# evaluate_xN_score(5, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_11_27')
# evaluate_xN_score(6, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, 'square_11_27')




# 1. L2 Score
# 2. Frequency Detection
# 3. Graph of score for each method