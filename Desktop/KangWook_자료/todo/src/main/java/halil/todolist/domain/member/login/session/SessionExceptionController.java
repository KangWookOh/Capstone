package halil.todolist.domain.member.login.session;

import halil.todolist.domain.member.exception.session.EmailDuplicate;
import halil.todolist.domain.member.exception.session.ErrorResponse;
import halil.todolist.domain.member.exception.session.LoginUserNotFound;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
public class SessionExceptionController {

    @ExceptionHandler(LoginUserNotFound.class)
    @ResponseStatus(HttpStatus.NOT_FOUND)
    public ResponseEntity<ErrorResponse> LoginUserNotFound(LoginUserNotFound e) {
        ErrorResponse response = ErrorResponse.builder()
                .message(e.getMessage())
                .build();

        return ResponseEntity.badRequest().body(response);
    }

    @ExceptionHandler(EmailDuplicate.class)
    @ResponseStatus(HttpStatus.NOT_FOUND)
    public ResponseEntity<ErrorResponse> emailDuplicate(EmailDuplicate e) {
        ErrorResponse response = ErrorResponse.builder()
                .message(e.getMessage())
                .build();

        return ResponseEntity.badRequest().body(response);
    }
}
