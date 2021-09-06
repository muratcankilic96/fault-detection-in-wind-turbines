#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include "dsp.h"
#include <QMainWindow>
#include <QLineEdit>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    std::vector<fvec_t *> read_wav_gui(fvec_t * vec, QLineEdit * line, uint_t * sample_rate);
    std::vector<fmat_t *> create_mfcc_gui(std::vector<fvec_t *> vec, uint_t sample_rate);
    std::vector<fmat_t *> create_spectrogram_gui(std::vector<fvec_t *> vec);
    std::vector<fmat_t *> create_mel_spectrogram_gui(std::vector<fvec_t *> vec);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    fvec_t * working_wav, * problematic_wav, * not_working_wav;
    std::vector<fmat_t *> working_mfcc, problematic_mfcc, not_working_mfcc;
    std::vector<fmat_t *> working_spectrogram, problematic_spectrogram, not_working_spectrogram;
    std::vector<fmat_t *> working_mel_spectrogram, problematic_mel_spectrogram, not_working_mel_spectrogram;
    std::vector<fvec_t *> working_mat, problematic_mat, not_working_mat;
    uint_t working_sample_rate, problematic_sample_rate, not_working_sample_rate;
    void file_select_button_clicked(QLineEdit * line);
    void derive_data_button_clicked();
};
#endif // MAINWINDOW_H
