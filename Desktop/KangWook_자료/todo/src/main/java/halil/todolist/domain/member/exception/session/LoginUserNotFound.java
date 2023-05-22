package halil.todolist.domain.member.exception.session;

import lombok.Getter;

@Getter
public class LoginUserNotFound extends RuntimeException {

    private static final String MESSAGE = "일치하는 사용자가 없습니다.";

    public LoginUserNotFound() {
        super(MESSAGE);
    }
}
