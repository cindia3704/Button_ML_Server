from django.db import models

# # Create your models here.
# from django.db import models
# from django.contrib.auth.models import (AbstractBaseUser, BaseUserManager)

# class UserManager(BaseUserManager):
#     def create_user(self, userEmail, is_staff=False, is_admin=False, is_active=True, confirmedEmail=False, password=None):
#         if not userEmail:
#             raise ValueError("Users must have an email address")
#         if not password:
#             raise ValueError("Users must have a password")
#         user_obj = self.model(
#             userEmail=self.normalize_email(userEmail)
#         )
#         user_obj.set_password(password)
#         user_obj.staff = is_staff
#         user_obj.admin = is_admin
#         user_obj.active = is_active
#         user_obj.confirmedEmail = confirmedEmail
#         user_obj.save(using=self._db)
#         return user_obj

#     def create_superuser(self, userEmail, password=None):
#         user = self.create_user(
#             userEmail=userEmail,
#             password=password,
#             is_staff=True,
#             is_admin=True,
#             confirmedEmail=False,
#         )
#         return user


# class User(AbstractBaseUser):
#     userEmail = models.EmailField(
#         max_length=255, unique=True, verbose_name="User Email", default='a@gmail.com')
#     active = models.BooleanField(default=True)
#     admin = models.BooleanField(default=False)
#     staff = models.BooleanField(default=False)
#     confirmedEmail = models.BooleanField(default=False)
#     # friendlist = models.ManyToManyField(
#     #     'Friend', related_name="friend_users", blank=True)
#     USERNAME_FIELD = 'userEmail'
#     # email & password = required by default
#     REQUIRED_FIELDS = ['']
#     objects = UserManager()

#     def __str__(self):
#         return self.userEmail

#     def has_perm(self, perm, obj=None):
#         return True

#     def has_module_perms(self, app_label):
#         return True

#     # def get_gender(self):
#     #     return self.userGender

#     # def get_nickname(self):
#     #     return self.userNickName

#     def get_email(self):
#         return self.userEmail

#     # def get_friends(self):
#     #     return self.friends

#     @property
#     def is_staff(self):
#         return self.staff

#     def is_admin(self):
#         return self.admin

#     def is_active(self):
#         return self.active


# class buttonUser(models.Model):
#     userEmail = models.EmailField(
#         max_length=255, unique=True, verbose_name="User Email", default='a@gmail.com')
#     password = models.CharField(max_length=63, verbose_name="password")
#     registered_dttm = models.DateTimeField(
#         auto_now_add=True, verbose_name="registered date")

#     class Meta:
#         db_table = 'buttonuser'
class Cloth_S(models.Model):
    id = models.AutoField(primary_key=True,
                          verbose_name="closetID",
                          unique=True)
    photo = models.ImageField(
        default='button/media/default.jpg', null=True, blank=True)
