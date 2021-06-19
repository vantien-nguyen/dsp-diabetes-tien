# dsp-diabetes-tien


By Van Tien NGUYEN

1.  This project uses the diabetes dataset following the instruction: https://github.com/EPITA-courses/dsp_practical_work/blob/master/instructions/pw4.md
2.  My Application includes 2 main folders:
  - frontend: streamlit application
  - backend: FastAPI - APIs:
	+ GET  /models: get all available models
	+ POST /training: trainng the model with data: return rmse, mae, r2
	+ POST /predict: predict one patient
	+ POST /predict_obj predict list of patients
  - diabetes datasets are saved in directory: backend/data
  - models are save in: backend/models
3. Runing:
  - clone project: git clone git@github.com:tiennguyenhust/dsp-diabetes-tien.git
  - move to the root directory: cd dsp-diabetes-tien
  - create the environment using requirements-dev.txt file
  - Run Backend: python backend/main.py
  - Run Frontend: streamlit run frontend/app.py

4. Deployment on Heroku:
  - Frontend: https://dspdiabetes.herokuapp.com/
  - Backend: https://dspdiabetesapi.herokuapp.com/docs
  * Because The Heroku filesystem is ephemeral, so all saved models are not kept. That means the API get models doesn't work, and api training model cannot save the model

==> Solution: AWS S3 or Database (postgresql ie.) but it's not problem in this project!
