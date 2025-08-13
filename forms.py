from flask_wtf import FlaskForm
from wtforms import StringField,PasswordField,SubmitField,ValidationError
from wtforms.validators import Length,EqualTo,Email,DataRequired
from backend.credentials import User


class RegisterForm(FlaskForm):

    def valid_username(self, username_to_check):
        user=User.query.filter_by(username=username_to_check.data).first()
        if user:
            raise ValidationError('Username already exists. Please choose a different one.')


    def valid_email(self, email_to_check):
        email=User.query.filter_by(email=email_to_check.data).first()

        if email:
            raise ValidationError('Email already exists. Please choose a different one.')

    username=StringField(label='Username',validators=[Length(min=5, max=20),DataRequired()])
    email_address = StringField(label='Email', validators=[Email(), DataRequired()])
    pswd=PasswordField(label='Password',validators=[Length(min=8, max=20),DataRequired()])
    confirm_pswd=PasswordField(label='Confirm Password',validators=[EqualTo('pswd'),DataRequired()])
    submit=SubmitField(label='Register')

class LoginForm(FlaskForm):
    username=StringField(label='Username', validators=[DataRequired()])
    pswd=PasswordField(label='Password', validators=[DataRequired()])
    submit=SubmitField(label='Login')