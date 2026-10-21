import os



import io



import pytest



from pathlib import Path



from backend.services.transcription import TranscriptionService, TranscriptionError



from backend.config import settings











def test_transcribe_audio_endpoint_success(client):



    """Test POST /transcribe-audio returns a valid transcript and metadata without auto-committing."""



    fake_audio_bytes = b"RIFF....WAVEfmt ...." + b"\x00" * 1024



    files = {"file": ("symptoms.wav", fake_audio_bytes, "audio/wav")}



    params = {"language": "English", "model_size": "base"}







    response = client.post("/transcribe-audio", files=files, params=params)



    assert response.status_code == 200



    res_data = response.json()







    assert "transcript" in res_data



    assert len(res_data["transcript"]) > 0



    assert "detected_language" in res_data



    assert "confidence" in res_data



    assert res_data["confidence"] > 0.0



    assert res_data["model_size"] == "base"



    assert res_data["is_mock"] is True











def test_transcribe_audio_endpoint_hindi(client):



    """Test POST /transcribe-audio handles Hindi language parameter."""



    fake_audio_bytes = b"RIFF....WAVEfmt ...." + b"\x00" * 512



    files = {"file": ("hindi_speech.mp3", fake_audio_bytes, "audio/mpeg")}



    params = {"language": "Hindi", "model_size": "tiny"}







    response = client.post("/transcribe-audio", files=files, params=params)



    assert response.status_code == 200



    res_data = response.json()







    assert res_data["detected_language"] in ["hi", "Hindi"]



    assert res_data["model_size"] == "tiny"











def test_transcribe_audio_endpoint_empty_file(client):



    """Test POST /transcribe-audio rejects empty audio files with 400."""



    files = {"file": ("empty.wav", b"", "audio/wav")}



    response = client.post("/transcribe-audio", files=files)



    assert response.status_code == 400



    assert "empty" in response.json()["detail"].lower()











def test_transcribe_audio_endpoint_invalid_format(client):



    """Test POST /transcribe-audio rejects unsupported audio formats."""



    files = {"file": ("malicious.exe", b"MZ......", "application/octet-stream")}



    response = client.post("/transcribe-audio", files=files)



    assert response.status_code == 400



    assert "unsupported" in response.json()["detail"].lower()











def test_transcribe_service_cleanup_and_tempfile():



    """Verify transcription service cleans up temporary audio files safely."""



    service = TranscriptionService()



    fake_audio = b"RIFFFAKEAUDIOBYTES" * 50



    temp_dir = Path(settings.CAREFLOW_TEMP_DIR)







    # Count files in temp_dir before



    before_files = set(temp_dir.glob("careflow_audio_*"))







    result = service.transcribe_audio(



        file_bytes=fake_audio,



        filename="test_recording.wav",



        language="English",



        model_size="base"



    )







    # Count files in temp_dir after



    after_files = set(temp_dir.glob("careflow_audio_*"))







    # Assert no leaked files



    assert after_files == before_files



    assert result.confidence is not None



    assert result.confidence > 0.5











def test_transcribe_service_size_limit_rejection():



    """Verify transcription service enforces maximum file size limit."""



    service = TranscriptionService()



    huge_bytes = b"0" * (26 * 1024 * 1024)  # 26 MB







    with pytest.raises(TranscriptionError) as exc_info:



        service.transcribe_audio(



            file_bytes=huge_bytes,



            filename="large_file.wav"



        )



    assert "exceeds" in str(exc_info.value).lower()











def test_hosted_transcription_groq_and_hf_mocked():



    """Verify hosted transcription parses Groq / OpenAI compatible and Hugging Face responses."""



    from unittest.mock import patch, MagicMock







    service = TranscriptionService()



    fake_audio = b"RIFFFAKEAUDIOBYTES"







    # Test Groq / OpenAI compatible response



    with patch.object(settings, "WHISPER_ENDPOINT", "https://api.groq.com/openai/v1/audio/transcriptions"):



        with patch.object(settings, "WHISPER_API_KEY", "gsk_test_api_key"):



            with patch("requests.post") as mock_post:



                mock_resp = MagicMock()



                mock_resp.status_code = 200



                mock_resp.json.return_value = {"text": "Mujhe do din se bukhar aur ulti hai"}



                mock_post.return_value = mock_resp







                res = service._hosted_transcribe(



                    file_bytes=fake_audio,



                    filename="voice.wav",



                    language="Hindi"



                )



                assert res is not None



                assert "bukhar" in res.transcript



                assert res.is_mock is False



                assert res.detected_language == "hi"







    # Test Hugging Face response



    with patch.object(settings, "WHISPER_ENDPOINT", "https://api-inference.huggingface.co/models/openai/whisper-large-v3"):



        with patch.object(settings, "WHISPER_API_KEY", "hf_test_token"):



            with patch("requests.post") as mock_post:



                mock_resp = MagicMock()



                mock_resp.status_code = 200



                mock_resp.json.return_value = {"text": "I have severe chest pressure"}



                mock_post.return_value = mock_resp







                res = service._hosted_transcribe(



                    file_bytes=fake_audio,



                    filename="voice.wav",



                    language="English"



                )



                assert res is not None



                assert "chest pressure" in res.transcript



                assert res.model_size == "hosted-huggingface"







    # Test auto-routing when WHISPER_ENDPOINT is Groq but user provides an hf_ token in WHISPER_API_KEY



    with patch.object(settings, "WHISPER_ENDPOINT", "https://api.groq.com/openai/v1/audio/transcriptions"):



        with patch.object(settings, "WHISPER_API_KEY", "hf_my_secret_token"):



            with patch("requests.post") as mock_post:



                mock_resp = MagicMock()



                mock_resp.status_code = 200



                mock_resp.json.return_value = [{"text": "Automatic routing worked"}]



                mock_post.return_value = mock_resp







                res = service._hosted_transcribe(



                    file_bytes=fake_audio,



                    filename="voice.wav",



                    language="English"



                )



                assert res is not None



                assert res.transcript == "Automatic routing worked"



                # Check that request was sent to router.huggingface.co instead of api.groq.com



                call_url = mock_post.call_args[0][0]



                assert "router.huggingface.co" in call_url







def test_deepgram_transcription_success():



    """Verify Deepgram STT (nova-2, en-IN) parses channels, words, duration, and smart formatting."""



    from unittest.mock import patch, MagicMock







    service = TranscriptionService()



    fake_audio = b"RIFFFAKEAUDIOBYTES"







    mock_resp = MagicMock()



    mock_resp.results.channels[0].alternatives[0].transcript = "mujhe 2 din se chest pain aur fever hai"



    mock_resp.results.channels[0].alternatives[0].confidence = 0.98



    mock_resp.results.channels[0].detected_language = "en-IN"



    mock_resp.metadata.duration = 4.25



    



    mock_word = MagicMock()



    mock_word.word = "chest"



    mock_word.start = 1.1



    mock_word.end = 1.5



    mock_word.confidence = 0.99



    mock_resp.results.channels[0].alternatives[0].words = [mock_word]







    with patch.object(settings, "USE_DEEPGRAM", True):



        with patch.object(settings, "DEEPGRAM_API_KEY", "dg_test_key_abc123"):



            with patch("deepgram.DeepgramClient") as MockDGClient:



                mock_client_instance = MagicMock()



                MockDGClient.return_value = mock_client_instance



                mock_client_instance.listen.rest.v.return_value.transcribe_file.return_value = mock_resp







                res = service.transcribe_audio(



                    file_bytes=fake_audio,



                    filename="symptoms.wav",



                    language="en-IN"



                )







                assert res is not None



                assert res.transcript == "mujhe 2 din se chest pain aur fever hai"



                assert res.confidence == 0.98



                assert res.duration_seconds == 4.25



                assert res.model_size == "deepgram-nova-2"



                assert res.is_mock is False



                assert len(res.segments) == 1



                assert res.segments[0]["text"] == "chest"



                



                # Verify SDK v3 listen.rest.v("1").transcribe_file was invoked



                mock_client_instance.listen.rest.v.assert_called_with("1")



                call_args = mock_client_instance.listen.rest.v.return_value.transcribe_file.call_args



                options = call_args[0][1]



                assert options.model == "nova-2"



                assert options.language == "en-IN"



                assert options.smart_format is True



                assert options.punctuate is True











def test_deepgram_translation_support():



    """Verify Deepgram STT passes translation addons when translate_to_english is enabled."""



    from unittest.mock import patch, MagicMock







    service = TranscriptionService()



    fake_audio = b"RIFFFAKEAUDIOBYTES"







    mock_resp = MagicMock()



    mock_resp.results.channels[0].alternatives[0].transcript = "I have had chest pain and fever for two days."



    mock_resp.results.channels[0].alternatives[0].confidence = 0.97



    mock_resp.results.channels[0].detected_language = "en"



    mock_resp.metadata.duration = 3.8



    mock_resp.results.channels[0].alternatives[0].words = []







    with patch.object(settings, "USE_DEEPGRAM", True):



        with patch.object(settings, "DEEPGRAM_API_KEY", "dg_test_key_abc123"):



            with patch("deepgram.DeepgramClient") as MockDGClient:



                mock_client_instance = MagicMock()



                MockDGClient.return_value = mock_client_instance



                mock_client_instance.listen.rest.v.return_value.transcribe_file.return_value = mock_resp







                res = service.transcribe_audio(



                    file_bytes=fake_audio,



                    filename="symptoms.wav",



                    translate_to_english=True



                )







                assert res is not None



                assert "chest pain" in res.transcript



                call_kwargs = mock_client_instance.listen.rest.v.return_value.transcribe_file.call_args[1]



                assert call_kwargs.get("addons") == {"translate": "en"}











def test_deepgram_feature_flag_fallback():



    """Verify that setting USE_DEEPGRAM=False skips Deepgram and uses fallback."""



    from unittest.mock import patch







    service = TranscriptionService()



    fake_audio = b"RIFFFAKEAUDIOBYTES"







    with patch.object(settings, "USE_DEEPGRAM", False):



        with patch.object(settings, "ASR_MODE", "mock"):



            res = service.transcribe_audio(



                file_bytes=fake_audio,



                filename="cardiac_case.wav"



            )



            assert res is not None



            assert res.is_mock is True



            assert "crushing chest pain" in res.transcript











def test_deepgram_tts_synthesis():



    """Verify Deepgram Text-to-Speech synthesis stream."""



    from unittest.mock import patch, MagicMock







    service = TranscriptionService()



    fake_mp3 = b"ID3\x04\x00\x00\x00" + b"\x00" * 128







    mock_resp = MagicMock()



    mock_resp.status_code = 200



    mock_resp.content = fake_mp3







    with patch.object(settings, "DEEPGRAM_API_KEY", "dg_test_key_abc123"):



        with patch("deepgram.DeepgramClient") as MockDGClient:



            mock_client_instance = MagicMock()



            MockDGClient.return_value = mock_client_instance



            mock_client_instance.speak.rest.v.return_value.stream_raw.return_value = mock_resp







            audio_data = service.text_to_speech("Patient reports severe pain in chest.")



            assert audio_data == fake_mp3



            mock_client_instance.speak.rest.v.assert_called_with("1")



def test_deepgram_v7_sdk_compatibility():

    """Verify Deepgram STT works when client exposes v7 SDK interface (client.listen.v1.media)."""

    from unittest.mock import patch, MagicMock



    service = TranscriptionService()

    fake_audio = b"RIFFFAKEAUDIOBYTES"



    mock_resp = MagicMock()

    mock_resp.results.channels[0].alternatives[0].transcript = "mujhe bukhar aur khansi hai"

    mock_resp.results.channels[0].alternatives[0].confidence = 0.99

    mock_resp.results.channels[0].detected_language = "en-IN"

    mock_resp.metadata.duration = 2.5

    mock_resp.results.channels[0].alternatives[0].words = []



    with patch.object(settings, "USE_DEEPGRAM", True):

        with patch.object(settings, "DEEPGRAM_API_KEY", "dg_test_key_v7"):

            with patch("deepgram.DeepgramClient") as MockDGClient:

                mock_client = MagicMock()

                # Simulate v7 structure (no client.listen.rest)

                del mock_client.listen.rest

                mock_client.listen.v1.media.transcribe_file.return_value = mock_resp

                MockDGClient.return_value = mock_client



                res = service.transcribe_audio(

                    file_bytes=fake_audio,

                    filename="voice.wav",

                    language="en-IN"

                )



                assert res is not None

                assert res.transcript == "mujhe bukhar aur khansi hai"

                assert res.confidence == 0.99

                mock_client.listen.v1.media.transcribe_file.assert_called_once()

def test_deepgram_failure_raises_honest_error_not_runny_nose():
    """Verify that if Deepgram fails or detects no speech, it raises TranscriptionError and NEVER returns fake runny nose."""
    from unittest.mock import patch, MagicMock

    service = TranscriptionService()
    fake_audio = b"RIFFFAKEAUDIOBYTES"

    # Simulate Deepgram returning empty transcript
    mock_resp = MagicMock()
    mock_resp.results.channels[0].alternatives[0].transcript = ""
    mock_resp.results.channels[0].alternatives[0].words = []

    with patch.object(settings, "USE_DEEPGRAM", True):
        with patch.object(settings, "DEEPGRAM_API_KEY", "dg_test_key_abc123"):
            with patch("deepgram.DeepgramClient") as MockDGClient:
                mock_client = MagicMock()
                mock_client.listen.rest.v.return_value.transcribe_file.return_value = mock_resp
                MockDGClient.return_value = mock_client

                with pytest.raises(TranscriptionError) as exc_info:
                    service.transcribe_audio(
                        file_bytes=fake_audio,
                        filename="recording.wav",
                        language="en"
                    )

                # Ensure error message clearly states problem in recording the voice
                assert "Problem in recording the voice" in str(exc_info.value)
                # Ensure fake runny nose symptom is NEVER present
                assert "runny nose" not in str(exc_info.value).lower()


def test_deepgram_cross_language_retry_auto_detect():
    """Verify that when explicit language returns empty, Deepgram automatically retries with detect_language=True."""
    from unittest.mock import patch, MagicMock

    service = TranscriptionService()
    fake_audio = b"RIFFFAKEAUDIOBYTES"

    # First call with explicit language returns empty
    mock_resp_empty = MagicMock()
    mock_resp_empty.results.channels[0].alternatives[0].transcript = ""

    # Second call with auto-detect returns English transcript
    mock_resp_success = MagicMock()
    mock_resp_success.results.channels[0].alternatives[0].transcript = "I have chest pain"
    mock_resp_success.results.channels[0].alternatives[0].confidence = 0.95
    mock_resp_success.results.channels[0].detected_language = "en"
    mock_resp_success.metadata.duration = 2.0
    mock_resp_success.results.channels[0].alternatives[0].words = []

    with patch.object(settings, "USE_DEEPGRAM", True):
        with patch.object(settings, "DEEPGRAM_API_KEY", "dg_test_key_abc123"):
            with patch("deepgram.DeepgramClient") as MockDGClient:
                mock_client = MagicMock()
                # Returns empty first, then returns success on auto-detect retry
                mock_client.listen.rest.v.return_value.transcribe_file.side_effect = [
                    mock_resp_empty,
                    mock_resp_success
                ]
                MockDGClient.return_value = mock_client

                res = service.transcribe_audio(
                    file_bytes=fake_audio,
                    filename="voice.wav",
                    language="Hindi"  # User selected Hindi, but spoke English
                )

                assert res is not None
                assert res.transcript == "I have chest pain"
                # transcribe_file should have been called twice (1: explicit hi, 2: auto-detect)
                assert mock_client.listen.rest.v.return_value.transcribe_file.call_count == 2

