package halil.todolist.domain.member.service;

import halil.todolist.domain.member.dto.SignUpDto;

public interface MemberService {
    public Long signUp(SignUpDto signUpDto);
    public String login(String email, String password);
    public void checkEmailDuplicate(String email);
}
